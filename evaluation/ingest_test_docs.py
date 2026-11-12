"""
Ingests the generated test documents into the RAG pipeline for evaluation.
This script runs the chunking + embedding pipeline locally (no server needed).
"""

import os
import sys
import asyncio

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Force in-memory cache for testing
os.environ.setdefault("CACHE_BACKEND", "memory")
os.environ.setdefault("CHROMA_PERSIST_DIR", os.path.join(os.path.dirname(__file__), "eval_chroma_data"))
os.environ.setdefault("BM25_DATA_DIR", os.path.join(os.path.dirname(__file__), "eval_bm25_data"))
os.environ.setdefault("OTEL_ENABLED", "false")
# Eval data is ingested and queried under one fixed owner, exactly like production
# requests (every retrieval path is owner-scoped).
EVAL_OWNER_ID = os.getenv("EVAL_OWNER_ID", "eval-owner")


async def ingest_documents():
    from app.rag.loaders.pdf_loader import load_pdf
    from app.rag.chunkers.recursive_chunker import chunk_documents
    from app.rag.embeddings.local_embedder import embed_documents
    from app.rag.vectorstore.chroma_store import add_documents
    from app.rag.vectorstore.bm25_store import get_bm25_store
    from app.core.cache import set_cache

    docs_dir = os.path.join(os.path.dirname(__file__), "test_documents")

    if not os.path.exists(docs_dir):
        print("ERROR: Test documents not found. Run generate_test_docs.py first.")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print("ERROR: No PDF files found in test_documents/")
        sys.exit(1)

    print(f"Found {len(pdf_files)} test documents to ingest.")

    all_chunks = []
    all_parent_stores = {}

    for filename in pdf_files:
        filepath = os.path.join(docs_dir, filename)
        print(f"  Loading: {filename}")

        docs = await load_pdf(filepath)
        for doc in docs:
            doc.metadata["filename"] = filename
            doc.metadata["doc_id"] = filename
            doc.metadata["owner_id"] = EVAL_OWNER_ID

        chunks, parent_store = chunk_documents(docs)
        all_chunks.extend(chunks)
        all_parent_stores.update(parent_store)
        print(f"    -> {len(chunks)} chunks, {len(parent_store)} parent segments")

    # Persist parent chunks durably (the eval runs in a separate process, so an
    # in-memory cache entry would be gone by then) and warm the cache.
    get_bm25_store().add_parents(all_parent_stores, doc_id="eval", owner_id=EVAL_OWNER_ID)
    for parent_id, parent_text in all_parent_stores.items():
        await set_cache(f"parent:{parent_id}", parent_text, ttl=None)

    # Embed all chunks
    texts = [c.page_content for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]

    print(f"\n  Embedding {len(texts)} chunks...")
    embeddings = await embed_documents(texts)

    if not embeddings:
        print("ERROR: Embedding generation failed.")
        sys.exit(1)

    # Store in ChromaDB
    print("  Storing in ChromaDB...")
    await add_documents(texts, embeddings, metadatas)

    # Store in BM25
    print("  Storing in BM25 index...")
    bm25_store = get_bm25_store()
    bm25_docs = []
    for i, text in enumerate(texts):
        doc_id = metadatas[i].get("hash", f"eval_{i}")
        source = metadatas[i].get("filename", "unknown")
        bm25_docs.append({"id": doc_id, "text": text, "doc_id": source, "owner_id": EVAL_OWNER_ID})
    bm25_store.add_documents(bm25_docs)

    print(f"\nIngestion complete: {len(texts)} chunks indexed.")


if __name__ == "__main__":
    asyncio.run(ingest_documents())
