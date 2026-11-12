import os
import uuid
from fastapi import APIRouter, UploadFile, File, Request, HTTPException, Depends, status, Form
from pydantic import BaseModel, HttpUrl, Field
from app.core.auth import get_current_user

from app.rag.vectorstore.chroma_store import delete_document, get_document_parents
from app.rag.vectorstore.bm25_store import get_bm25_store
from app.core.exceptions import DocumentTooLargeError
from app.core.cache import get_cache, delete_cache, bump_cache_generation
from app.core.upload import get_upload_dir
from app.core.rate_limit import limiter, UPLOAD_RATE_LIMIT, URL_RATE_LIMIT
import structlog
import asyncio

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


class URLUploadRequest(BaseModel):
    url: HttpUrl = Field(..., max_length=2048)
    replace: bool = True


@router.post("/upload")
@limiter.limit(UPLOAD_RATE_LIMIT)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    replace: bool = Form(True),
    user_id: int = Depends(get_current_user),
):
    max_size = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024

    safe_filename = os.path.basename(file.filename or "")
    if not safe_filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    import magic

    job_id = str(uuid.uuid4())
    owner_id = str(user_id)
    doc_id = f"{owner_id}:{uuid.uuid4()}"

    ext = os.path.splitext(safe_filename)[1].lower()
    upload_dir = get_upload_dir()
    target_path = os.path.join(upload_dir, f"{uuid.uuid4()}{ext}")

    try:
        total_size = 0
        with open(target_path, "wb") as f:
            while chunk := await file.read(8192):
                total_size += len(chunk)
                if total_size > max_size:
                    raise DocumentTooLargeError(
                        f"File exceeds maximum size of {max_size // (1024 * 1024)}MB"
                    )
                f.write(chunk)

        mime = magic.from_file(target_path, mime=True)
        allowed_mimes = {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        if mime not in allowed_mimes:
            if os.path.exists(target_path):
                os.remove(target_path)
            raise HTTPException(status_code=400, detail=f"Invalid file type detected: {mime}")

    except DocumentTooLargeError:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise e

    try:
        from app.worker import process_file_task
        process_file_task.delay(target_path, safe_filename, file.content_type or "", job_id, owner_id, doc_id, replace)
    except Exception as e:
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass
        logger.error("Failed to enqueue file processing task", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue unavailable. Please try again later.",
        )

    # R-12: Do NOT bump cache generation on enqueue; worker bumps only on successful ingestion.
    return {"job_id": job_id, "doc_id": doc_id, "status": "processing"}


@router.post("/url")
@limiter.limit(URL_RATE_LIMIT)
async def upload_url(
    body: URLUploadRequest,
    request: Request,
    user_id: int = Depends(get_current_user),
):
    job_id = str(uuid.uuid4())
    owner_id = str(user_id)
    doc_id = f"{owner_id}:{uuid.uuid4()}"
    try:
        from app.worker import process_url_task
        process_url_task.delay(str(body.url), job_id, owner_id, doc_id, body.replace)
    except Exception as e:
        logger.error("Failed to enqueue URL processing task", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue unavailable. Please try again later.",
        )
    # R-12: Do NOT bump cache generation on enqueue; worker bumps only on successful ingestion.
    return {"job_id": job_id, "doc_id": doc_id, "status": "processing"}


@router.get("")
async def get_documents(
    request: Request,
    user_id: int = Depends(get_current_user),
):
    owner_id = str(user_id)
    docs = await asyncio.to_thread(get_bm25_store().get_document_metadata, owner_id)
    return {"documents": docs}


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    request: Request,
    user_id: int = Depends(get_current_user),
):
    status = await get_cache(f"job:{job_id}")
    if not status:
        status = "unknown"
    return {"job_id": job_id, "status": status}


@router.delete("/{doc_id}")
async def delete_doc(
    doc_id: str,
    request: Request,
    user_id: int = Depends(get_current_user),
):
    owner_id = str(user_id)
    doc_meta = await asyncio.to_thread(get_bm25_store().get_document_metadata, None, doc_id)
    if not doc_meta or doc_meta[0].get("owner_id") != owner_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Document does not belong to the current user",
        )

    # Clean up parent chunks from Redis and durable sparse store (R-12)
    parents = await get_document_parents(doc_id, owner_id=owner_id)
    store_parents = await asyncio.to_thread(get_bm25_store().get_parents_by_doc_id, doc_id, owner_id)
    for p_id in set(parents).union(store_parents):
        await delete_cache(f"parent:{p_id}")
    await asyncio.to_thread(get_bm25_store().delete_parents_by_doc_id, doc_id, owner_id)
        
    # Transactional safety is handled best-effort via execution ordering
    await delete_document(doc_id, owner_id=owner_id)
    await asyncio.to_thread(get_bm25_store().delete_documents_by_doc_id, doc_id, owner_id=owner_id)
    
    # Clean up knowledge graph
    try:
        if os.getenv("DISABLE_GRAPH", "0") != "1":
            from app.rag.graph.neo4j_manager import get_neo4j_manager
            await asyncio.to_thread(
                get_neo4j_manager().delete_document_graph,
                doc_id,
                owner_id,
            )
    except Exception as e:
        logger.warning(f"Graph deletion failed for doc {doc_id} (non-fatal): {e}")
    
    logger.info("Document deleted", doc_id=doc_id)
    await bump_cache_generation(owner_id)
    return {"status": "deleted"}
