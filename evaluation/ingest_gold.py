"""
Owner-scoped ingestion for the answer-quality evaluation.

Ingests the 3 synthetic PDFs under a FIXED owner_id (EVAL_OWNER_ID) so the
harness exercises the same owner-scoped retrieval path the real app uses.

This replicates the SAFE lower-level ingestion path from
app.rag.ingestion.IngestionPipeline._process_docs:
  load -> chunk (parent/child) -> embed -> cache parents -> chroma add -> bm25 add
but WITH owner_id/parent_id/filename metadata set, and WITHOUT the LLM+Neo4j
graph-extraction step (which needs a running Neo4j and burns LLM calls).

Idempotent: Chroma add_documents dedups by hash id, and BM25 add_documents
upserts by id, so re-running does not duplicate chunks.
"""

import os
import sys
import asyncio

import eval_common  # noqa: F401  (sets local-mode env + sys.path before backend imports)
from eval_common import EVAL_OWNER_ID, TEST_DOCS_DIR


def ensure_test_documents() -> None:
    """Generate the synthetic PDFs if they are not already present."""
    needed = [
        "machine_learning_fundamentals.pdf",
        "python_programming_guide.pdf",
        "distributed_systems_overview.pdf",
    ]
    have_all = os.path.isdir(TEST_DOCS_DIR) and all(
        os.path.exists(os.path.join(TEST_DOCS_DIR, f)) for f in needed
    )
    if have_all:
        return
    print("Test documents missing — generating them with generate_test_docs.py ...")
    import generate_test_docs  # local module; writes into TEST_DOCS_DIR
    generate_test_docs.generate_ml_doc()
    generate_test_docs.generate_python_doc()
    generate_test_docs.generate_distributed_systems_doc()


async def ingest() -> int:
    from app.rag.loaders.pdf_loader import load_pdf
    from app.rag.chunkers.recursive_chunker import chunk_documents
    from app.rag.embeddings.local_embedder import embed_documents
    from app.rag.vectorstore.chroma_store import add_documents
    from app.rag.vectorstore.bm25_store import get_bm25_store
    from app.core.cache import set_cache
    from app.core.exceptions import ProcessingError

    ensure_test_documents()

    pdf_files = sorted(f for f in os.listdir(TEST_DOCS_DIR) if f.endswith(".pdf"))
    if not pdf_files:
        print("ERROR: no PDFs found in test_documents/ after generation.")
        return 0

    print(f"Ingesting {len(pdf_files)} documents under owner_id='{EVAL_OWNER_ID}'...")

    total_chunks = 0
    bm25_store = get_bm25_store()

    for filename in pdf_files:
        filepath = os.path.join(TEST_DOCS_DIR, filename)
        doc_id = f"{EVAL_OWNER_ID}:{filename}"

        docs = await load_pdf(filepath)
        for doc in docs:
            doc.metadata["filename"] = filename
            doc.metadata["doc_id"] = doc_id
            doc.metadata["owner_id"] = EVAL_OWNER_ID

        chunks, parent_store = chunk_documents(docs)
        if not chunks:
            print(f"  [{filename}] produced no chunks — skipping.")
            continue

        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        # Enforce the owner-scoped metadata the retriever filters on (defense-in-depth).
        for m in metadatas:
            m["doc_id"] = doc_id
            m["filename"] = filename
            m["owner_id"] = EVAL_OWNER_ID

        embeddings = await embed_documents(texts)
        if not embeddings or len(embeddings) != len(texts):
            raise ProcessingError(f"embedding generation failed for {filename}")

        # Parent chunks -> cache (hybrid_retriever hydrates parent:{parent_id}).
        for parent_id, parent_text in parent_store.items():
            await set_cache(f"parent:{parent_id}", parent_text, ttl=None)

        await add_documents(texts, embeddings, metadatas)

        bm25_docs = [
            {
                "id": metadatas[i].get("hash", f"{doc_id}_{i}"),
                "text": texts[i],
                "doc_id": doc_id,
                "owner_id": EVAL_OWNER_ID,
            }
            for i in range(len(texts))
        ]
        bm25_store.add_documents(bm25_docs)

        import datetime
        from datetime import timezone
        bm25_store.update_document_metadata(
            doc_id=doc_id,
            filename=filename,
            chunk_count=len(texts),
            status="completed",
            created_at=datetime.datetime.now(timezone.utc).isoformat(),
            owner_id=EVAL_OWNER_ID,
        )

        total_chunks += len(texts)
        print(f"  [{filename}] -> {len(texts)} chunks, {len(parent_store)} parent segments")

    print(f"\nIngestion complete: {total_chunks} chunks indexed under owner_id='{EVAL_OWNER_ID}'.")
    return total_chunks


if __name__ == "__main__":
    asyncio.run(ingest())
