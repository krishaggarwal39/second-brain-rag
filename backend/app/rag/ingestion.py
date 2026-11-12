import os
import uuid
import asyncio
import datetime
from datetime import timezone
import structlog
from typing import List, Optional
from langchain_core.documents import Document

from app.rag.loaders.pdf_loader import load_pdf
from app.rag.loaders.docx_loader import load_docx
from app.rag.loaders.web_loader import load_web
from app.rag.chunkers.recursive_chunker import chunk_documents
from app.rag.embeddings.local_embedder import embed_documents
from app.rag.vectorstore.chroma_store import (
    add_documents,
    delete_document,
    get_document_parents,
)
from app.rag.vectorstore.bm25_store import get_bm25_store
from app.core.exceptions import UnsupportedFormatError, ProcessingError, DocumentTooLargeError
from app.core.cache import set_cache, delete_cache, bump_cache_generation

logger = structlog.get_logger(__name__)

# Permanent errors: bad file, unsupported format, empty/no-text, oversized.
# These terminate immediately with failed: <msg> and are not retried by Celery (R-18).
PERMANENT_ERRORS = (UnsupportedFormatError, ProcessingError, DocumentTooLargeError)


class IngestionPipeline:
    @staticmethod
    async def process_file(
        file_path: str,
        filename: str,
        content_type: str,
        job_id: str,
        owner_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        replace: bool = True,
    ):
        await set_cache(f"job:{job_id}", "processing")
        if not doc_id:
            doc_id = f"{owner_id}:{uuid.uuid4()}" if owner_id else str(uuid.uuid4())

        data_modified = False
        try:
            content_type_lower = content_type.lower()
            filename_lower = filename.lower()

            if "pdf" in content_type_lower or filename_lower.endswith(".pdf"):
                docs = await load_pdf(file_path)
            elif "word" in content_type_lower or "docx" in content_type_lower or filename_lower.endswith(".docx"):
                docs = await load_docx(file_path)
            else:
                raise UnsupportedFormatError(f"Unsupported format for {filename}")

            # R-20: If extraction yields no documents, raise permanent ProcessingError
            if not docs or not any(d.page_content.strip() for d in docs):
                raise ProcessingError("No extractable text found")

            for doc in docs:
                doc.metadata["filename"] = filename
                doc.metadata["doc_id"] = doc_id
                if owner_id:
                    doc.metadata["owner_id"] = str(owner_id)

            data_modified = await IngestionPipeline._process_docs(
                docs=docs,
                job_id=job_id,
                doc_id=doc_id,
                filename=filename,
                owner_id=owner_id,
                replace=replace,
            )

            await set_cache(f"job:{job_id}", "completed")
            if owner_id:
                await bump_cache_generation(str(owner_id))

            # Success: clean up staging upload file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass

        except PERMANENT_ERRORS as e:
            # R-18: Permanent error -> record failed: <message> with no retry and delete file
            logger.error(f"Permanent ingestion failure for file {filename}: {e}")
            await set_cache(f"job:{job_id}", f"failed: {str(e)}")
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass
            return

        except Exception as e:
            # R-18: Transient error (network, Redis, DB connection) -> re-raise so Celery retries.
            # Keep the upload file until the last attempt.
            logger.error(f"Transient ingestion error for file {filename}: {e}")
            if data_modified and owner_id:
                try:
                    await bump_cache_generation(str(owner_id))
                except Exception:
                    pass
            raise

    @staticmethod
    async def process_url(
        url: str,
        job_id: str,
        owner_id: Optional[str] = None,
        doc_id: Optional[str] = None,
        replace: bool = True,
    ):
        await set_cache(f"job:{job_id}", "processing")
        if not doc_id:
            doc_id = f"{owner_id}:{uuid.uuid4()}" if owner_id else str(uuid.uuid4())

        data_modified = False
        try:
            docs = await load_web(url)
            # R-20: If extraction yields no documents, raise permanent ProcessingError
            if not docs or not any(d.page_content.strip() for d in docs):
                raise ProcessingError("No extractable text found")

            for doc in docs:
                doc.metadata["filename"] = url
                doc.metadata["doc_id"] = doc_id
                if owner_id:
                    doc.metadata["owner_id"] = str(owner_id)

            data_modified = await IngestionPipeline._process_docs(
                docs=docs,
                job_id=job_id,
                doc_id=doc_id,
                filename=url,
                owner_id=owner_id,
                replace=replace,
            )

            await set_cache(f"job:{job_id}", "completed")
            if owner_id:
                await bump_cache_generation(str(owner_id))

        except PERMANENT_ERRORS as e:
            logger.error(f"Permanent ingestion failure for URL {url}: {e}")
            await set_cache(f"job:{job_id}", f"failed: {str(e)}")
            return

        except Exception as e:
            logger.error(f"Transient ingestion error for URL {url}: {e}")
            if data_modified and owner_id:
                try:
                    await bump_cache_generation(str(owner_id))
                except Exception:
                    pass
            raise

    @staticmethod
    async def _process_docs(
        docs: List[Document],
        job_id: str,
        doc_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        filename: Optional[str] = None,
        source_id: Optional[str] = None,
        replace: bool = True,
    ) -> bool:
        """Process documents into parent/child chunks, embeddings, vectorstore, and BM25.

        Returns True if data was modified.
        """
        if doc_id is None and source_id is not None:
            doc_id = source_id
        if not doc_id:
            doc_id = f"{owner_id}:{uuid.uuid4()}" if owner_id else str(uuid.uuid4())
        if not filename:
            filename = source_id or doc_id

        chunks, parent_store = chunk_documents(docs)
        # R-20: If extraction yields no chunks, raise permanent ProcessingError
        if not chunks:
            raise ProcessingError("No extractable text found")

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

        # R-19: Identify old versions before writing new version
        old_doc_ids = []
        if replace and owner_id and filename:
            old_docs = await asyncio.to_thread(
                get_bm25_store().get_document_metadata, owner_id=str(owner_id)
            )
            if old_docs:
                for old_doc in old_docs:
                    if not isinstance(old_doc, dict):
                        continue
                    old_id = old_doc.get("doc_id")
                    if (
                        old_doc.get("filename") == filename
                        and old_id
                        and old_id != doc_id
                    ):
                        old_doc_ids.append(old_id)

        # R-19: Write new version FIRST under new doc_id (vectors, BM25, graph, parents)
        new_data_written = False
        try:
            # (1) Durable parent chunks (R-12) & Redis cache
            if parent_store:
                await asyncio.to_thread(
                    get_bm25_store().add_parents,
                    parent_store,
                    doc_id,
                    str(owner_id) if owner_id else "",
                )
                for p_id, p_text in parent_store.items():
                    await set_cache(f"parent:{p_id}", p_text, ttl=None)
            new_data_written = True

            # (2) Chroma vectorstore
            await add_documents(texts, embeddings, metadatas)

            # (3) BM25 sparse store (R-25: wrapped in asyncio.to_thread)
            bm25_docs = []
            for i, text in enumerate(texts):
                chunk_id = metadatas[i].get("hash", f"{job_id}_{i}")
                bm25_docs.append({
                    "id": chunk_id,
                    "text": text,
                    "doc_id": doc_id,
                    "owner_id": str(owner_id) if owner_id else "",
                })

            await asyncio.to_thread(get_bm25_store().add_documents, bm25_docs)

            await asyncio.to_thread(
                get_bm25_store().update_document_metadata,
                doc_id=doc_id,
                filename=filename,
                chunk_count=len(texts),
                status="completed",
                created_at=datetime.datetime.now(timezone.utc).isoformat(),
                owner_id=owner_id,
            )

            # (4) Graph extraction
            try:
                max_segments = int(os.getenv("GRAPH_MAX_SEGMENTS", "5"))
            except (ValueError, TypeError):
                max_segments = 5
            max_segments = max(0, min(max_segments, 20))

            parent_candidates = list(parent_store.values()) if parent_store else texts
            graph_texts = parent_candidates[:max_segments]

            if graph_texts and os.getenv("DISABLE_GRAPH", "0") != "1":
                from app.rag.graph.graph_extractor import extract_and_store_graph

                sem = asyncio.Semaphore(2)

                async def bounded_extract(t: str):
                    async with sem:
                        try:
                            await extract_and_store_graph(
                                t,
                                owner_id=str(owner_id) if owner_id else None,
                                doc_id=doc_id,
                            )
                            await asyncio.sleep(1)
                        except Exception as e:
                            logger.warning(f"Graph extraction failed (non-fatal): {e}")

                try:
                    await asyncio.gather(*(bounded_extract(t) for t in graph_texts), return_exceptions=True)
                except Exception as e:
                    logger.warning(f"Graph extraction gather error (non-fatal): {e}")

        except Exception as exc:
            # R-19: On failure roll back what the new run wrote and keep the old version intact
            if new_data_written:
                # R-12: If ingestion fails after having modified data, bump cache generation
                if owner_id:
                    try:
                        await bump_cache_generation(str(owner_id))
                    except Exception:
                        pass
                logger.warning(f"Ingestion write failed; rolling back new doc_id={doc_id}: {exc}")
                try:
                    if parent_store:
                        for p_id in parent_store:
                            await delete_cache(f"parent:{p_id}")
                        await asyncio.to_thread(
                            get_bm25_store().delete_parents_by_doc_id,
                            doc_id,
                            str(owner_id) if owner_id else None,
                        )
                    await delete_document(doc_id, owner_id=str(owner_id) if owner_id else None)
                    await asyncio.to_thread(
                        get_bm25_store().delete_documents_by_doc_id,
                        doc_id,
                        owner_id=str(owner_id) if owner_id else None,
                    )
                    if os.getenv("DISABLE_GRAPH", "0") != "1":
                        from app.rag.graph.neo4j_manager import get_neo4j_manager
                        await asyncio.to_thread(
                            get_neo4j_manager().delete_document_graph,
                            doc_id,
                            str(owner_id) if owner_id else None,
                        )
                except Exception as rb_err:
                    logger.error(f"Rollback failed for {doc_id}: {rb_err}")
            raise

        # R-19: Now that writing the new version has succeeded, delete the old version(s)
        for old_doc_id in old_doc_ids:
            logger.info(
                "Replacing old document version after successful new write",
                filename=filename,
                old_doc_id=old_doc_id,
                new_doc_id=doc_id,
                owner_id=str(owner_id),
            )
            # (1) Clean up old parents from Redis and DB
            try:
                parents = await get_document_parents(old_doc_id, owner_id=str(owner_id))
                store_parents = await asyncio.to_thread(
                    get_bm25_store().get_parents_by_doc_id, old_doc_id, str(owner_id)
                )
                for p_id in set(parents).union(store_parents):
                    await delete_cache(f"parent:{p_id}")
                await asyncio.to_thread(
                    get_bm25_store().delete_parents_by_doc_id, old_doc_id, str(owner_id)
                )
            except Exception as e:
                logger.warning(f"Old parent deletion error for {old_doc_id}: {e}")

            # (2) Chroma delete_document
            await delete_document(old_doc_id, owner_id=str(owner_id))

            # (3) BM25 delete_documents_by_doc_id
            await asyncio.to_thread(
                get_bm25_store().delete_documents_by_doc_id,
                old_doc_id,
                owner_id=str(owner_id),
            )

            # (4) Neo4j graph cleanup for old doc
            try:
                if os.getenv("DISABLE_GRAPH", "0") != "1":
                    from app.rag.graph.neo4j_manager import get_neo4j_manager
                    await asyncio.to_thread(
                        get_neo4j_manager().delete_document_graph,
                        old_doc_id,
                        str(owner_id),
                    )
            except Exception as e:
                logger.warning(
                    f"Graph deletion failed for old doc {old_doc_id} (non-fatal): {e}"
                )

        return True


ingestion_pipeline = IngestionPipeline()
