import os
import uuid
import asyncio
import structlog
from typing import List, Optional
from langchain_core.documents import Document

from app.rag.loaders.pdf_loader import load_pdf
from app.rag.loaders.docx_loader import load_docx
from app.rag.loaders.web_loader import load_web
from app.rag.chunkers.recursive_chunker import chunk_documents
from app.rag.embeddings.local_embedder import embed_documents
from app.rag.vectorstore.chroma_store import add_documents
from app.rag.vectorstore.bm25_store import get_bm25_store
from app.core.exceptions import UnsupportedFormatError, ProcessingError, DocumentTooLargeError
from app.core.cache import set_cache, bump_cache_generation

logger = structlog.get_logger(__name__)

class IngestionPipeline:
    @staticmethod
    async def process_file(
        file_path: str,
        filename: str,
        content_type: str,
        job_id: str,
        owner_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ):
        await set_cache(f"job:{job_id}", "processing")
        if not doc_id:
            doc_id = f"{owner_id}:{uuid.uuid4()}" if owner_id else str(uuid.uuid4())
        try:
            content_type_lower = content_type.lower()
            filename_lower = filename.lower()

            if "pdf" in content_type_lower or filename_lower.endswith(".pdf"):
                docs = await load_pdf(file_path)
            elif "word" in content_type_lower or "docx" in content_type_lower or filename_lower.endswith(".docx"):
                docs = await load_docx(file_path)
            else:
                raise UnsupportedFormatError(f"Unsupported format for {filename}")

            for doc in docs:
                doc.metadata["filename"] = filename
                doc.metadata["doc_id"] = doc_id
                if owner_id:
                    doc.metadata["owner_id"] = str(owner_id)

            await IngestionPipeline._process_docs(
                docs=docs,
                job_id=job_id,
                doc_id=doc_id,
                filename=filename,
                owner_id=owner_id,
            )

            await set_cache(f"job:{job_id}", "completed")
            if owner_id:
                await bump_cache_generation(str(owner_id))

        except (UnsupportedFormatError, ProcessingError, DocumentTooLargeError) as e:
            logger.error(f"Ingestion failed for file {filename}: {e}")
            await set_cache(f"job:{job_id}", f"failed: {str(e)}")
        except Exception as e:
            logger.error(f"Ingestion failed for file {filename}: {e}")
            await set_cache(f"job:{job_id}", "failed: An unexpected internal error occurred.")
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass

    @staticmethod
    async def process_url(
        url: str,
        job_id: str,
        owner_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ):
        await set_cache(f"job:{job_id}", "processing")
        if not doc_id:
            doc_id = f"{owner_id}:{uuid.uuid4()}" if owner_id else str(uuid.uuid4())
        try:
            docs = await load_web(url)
            for doc in docs:
                doc.metadata["filename"] = url
                doc.metadata["doc_id"] = doc_id
                if owner_id:
                    doc.metadata["owner_id"] = str(owner_id)
            await IngestionPipeline._process_docs(
                docs=docs,
                job_id=job_id,
                doc_id=doc_id,
                filename=url,
                owner_id=owner_id,
            )
            await set_cache(f"job:{job_id}", "completed")
            if owner_id:
                await bump_cache_generation(str(owner_id))
        except (UnsupportedFormatError, ProcessingError, DocumentTooLargeError) as e:
            logger.error(f"Ingestion failed for URL {url}: {e}")
            await set_cache(f"job:{job_id}", f"failed: {str(e)}")
        except Exception as e:
            logger.error(f"Ingestion failed for URL {url}: {e}")
            await set_cache(f"job:{job_id}", "failed: An unexpected internal error occurred.")

    @staticmethod
    async def _process_docs(
        docs: List[Document],
        job_id: str,
        doc_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        filename: Optional[str] = None,
        source_id: Optional[str] = None,
    ):
        if doc_id is None and source_id is not None:
            doc_id = source_id
        if not doc_id:
            doc_id = f"{owner_id}:{uuid.uuid4()}" if owner_id else str(uuid.uuid4())
        if not filename:
            filename = source_id or doc_id

        chunks, parent_store = chunk_documents(docs)
        if not chunks:
            return

        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        for m in metadatas:
            m["doc_id"] = doc_id
            if filename:
                m.setdefault("filename", filename)
            if owner_id:
                m["owner_id"] = str(owner_id)

        embeddings = await embed_documents(texts)
        if not embeddings or len(embeddings) != len(texts):
            raise ProcessingError("embedding generation failed")

        for p_id, p_text in parent_store.items():
            await set_cache(f"parent:{p_id}", p_text, ttl=None)

        await add_documents(texts, embeddings, metadatas)

        from app.rag.graph.graph_extractor import extract_and_store_graph

        bm25_docs = []
        for i, text in enumerate(texts):
            chunk_id = metadatas[i].get("hash", f"{job_id}_{i}")
            bm25_docs.append({
                "id": chunk_id,
                "text": text,
                "doc_id": doc_id,
                "owner_id": str(owner_id) if owner_id else "",
            })

        get_bm25_store().add_documents(bm25_docs)

        import datetime
        from datetime import timezone
        get_bm25_store().update_document_metadata(
            doc_id=doc_id,
            filename=filename,
            chunk_count=len(texts),
            status="completed",
            created_at=datetime.datetime.now(timezone.utc).isoformat(),
            owner_id=owner_id
        )

        try:
            max_segments = int(os.getenv("GRAPH_MAX_SEGMENTS", "5"))
        except (ValueError, TypeError):
            max_segments = 5
        max_segments = max(0, min(max_segments, 20))

        parent_candidates = list(parent_store.values()) if parent_store else texts
        graph_texts = parent_candidates[:max_segments]

        sem = asyncio.Semaphore(2)
        async def bounded_extract(t: str):
            async with sem:
                try:
                    await extract_and_store_graph(t)
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.warning(f"Graph extraction failed (non-fatal): {e}")

        if graph_texts:
            try:
                await asyncio.gather(*(bounded_extract(t) for t in graph_texts), return_exceptions=True)
            except Exception as e:
                logger.warning(f"Graph extraction gather error (non-fatal): {e}")

ingestion_pipeline = IngestionPipeline()
