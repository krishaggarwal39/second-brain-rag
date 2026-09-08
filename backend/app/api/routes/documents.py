import os
import uuid
import tempfile
from fastapi import APIRouter, UploadFile, File, Request, HTTPException, Depends, status
from pydantic import BaseModel, HttpUrl, Field
from app.core.auth import get_current_user

from app.rag.vectorstore.chroma_store import delete_document, get_document_parents
from app.rag.vectorstore.bm25_store import get_bm25_store
from app.core.exceptions import DocumentTooLargeError
from app.core.cache import get_cache, delete_cache, bump_cache_generation
from app.core.rate_limit import limiter, UPLOAD_RATE_LIMIT, URL_RATE_LIMIT
import structlog
import asyncio

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


class URLUploadRequest(BaseModel):
    url: HttpUrl = Field(..., max_length=2048)


@router.post("/upload")
@limiter.limit(UPLOAD_RATE_LIMIT)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
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

    fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(safe_filename)[1])
    try:
        total_size = 0
        with os.fdopen(fd, "wb") as f:
            while chunk := await file.read(8192):
                total_size += len(chunk)
                if total_size > max_size:
                    raise DocumentTooLargeError(
                        f"File exceeds maximum size of {max_size // (1024 * 1024)}MB"
                    )
                f.write(chunk)

        mime = magic.from_file(temp_path, mime=True)
        allowed_mimes = {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        if mime not in allowed_mimes:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Invalid file type detected: {mime}")

    except DocumentTooLargeError:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

    from app.worker import process_file_task
    process_file_task.delay(temp_path, safe_filename, file.content_type or "", job_id, owner_id, doc_id)
    await bump_cache_generation(owner_id)
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
    from app.worker import process_url_task
    process_url_task.delay(str(body.url), job_id, owner_id, doc_id)
    await bump_cache_generation(owner_id)
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

    # Clean up parent chunks from Redis
    parents = await get_document_parents(doc_id, owner_id=owner_id)
    for p_id in parents:
        await delete_cache(f"parent:{p_id}")
        
    # Transactional safety is handled best-effort via execution ordering
    await delete_document(doc_id, owner_id=owner_id)
    await asyncio.to_thread(get_bm25_store().delete_documents_by_doc_id, doc_id, owner_id=owner_id)
    
    logger.info("Document deleted", doc_id=doc_id)
    await bump_cache_generation(owner_id)
    return {"status": "deleted"}
