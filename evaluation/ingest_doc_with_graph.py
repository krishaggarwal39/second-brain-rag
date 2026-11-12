"""
Owner-scoped ingestion of a real document WITH Knowledge Graph extraction for evaluation.

Usage:
  python evaluation/ingest_doc_with_graph.py [path-to-file] [--name display_name.pdf]
  python evaluation/ingest_doc_with_graph.py

Ingests under EVAL_OWNER_ID using the same safe lower-level path as ingest_doc.py
and _process_docs (loader -> chunk -> embed -> cache parents -> chroma -> bm25),
and additionally populates the knowledge graph in Neo4j via extract_and_store_graph.
Uses bounded concurrency (asyncio.Semaphore(2)) and caps parent segments via
GRAPH_MAX_SEGMENTS (default: 15).
"""

import os
import sys
import argparse
import asyncio

# Ensure evaluation directory is in sys.path
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

import eval_common  # noqa: F401
from eval_common import EVAL_OWNER_ID, TEST_DOCS_DIR


async def ingest_with_graph(path: str, display_name: str) -> int:
    from app.rag.loaders.pdf_loader import load_pdf
    from app.rag.loaders.docx_loader import load_docx
    from app.rag.chunkers.recursive_chunker import chunk_documents
    from app.rag.embeddings.local_embedder import embed_documents
    from app.rag.vectorstore.chroma_store import add_documents
    from app.rag.vectorstore.bm25_store import get_bm25_store
    from app.rag.graph.graph_extractor import extract_and_store_graph
    from app.core.cache import set_cache
    from app.core.exceptions import ProcessingError

    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}")
        return 0

    lower = path.lower()
    if lower.endswith(".pdf"):
        docs = await load_pdf(path)
    elif lower.endswith(".docx"):
        docs = await load_docx(path)
    else:
        print("ERROR: unsupported file type (need .pdf or .docx)")
        return 0

    doc_id = f"{EVAL_OWNER_ID}:{display_name}"
    for d in docs:
        d.metadata["filename"] = display_name
        d.metadata["doc_id"] = doc_id
        d.metadata["owner_id"] = EVAL_OWNER_ID

    chunks, parent_store = chunk_documents(docs)
    if not chunks:
        print("ERROR: no chunks produced (document may be image-only / unextractable)")
        return 0

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    for m in metadatas:
        m["doc_id"] = doc_id
        m["filename"] = display_name
        m["owner_id"] = EVAL_OWNER_ID

    print(f"Embedding {len(texts)} chunks...")
    embeddings = await embed_documents(texts)
    if not embeddings or len(embeddings) != len(texts):
        raise ProcessingError("embedding generation failed")

    print(f"Caching {len(parent_store)} parent chunks...")
    for pid, ptext in parent_store.items():
        await set_cache(f"parent:{pid}", ptext, ttl=None)

    print("Adding documents to Chroma vector store...")
    await add_documents(texts, embeddings, metadatas)

    print("Adding documents to BM25 store...")
    bm25_store = get_bm25_store()
    bm25_store.add_documents([
        {"id": metadatas[i].get("hash", f"{doc_id}_{i}"),
         "text": texts[i], "doc_id": doc_id, "owner_id": EVAL_OWNER_ID}
        for i in range(len(texts))
    ])
    import datetime
    from datetime import timezone
    bm25_store.update_document_metadata(
        doc_id=doc_id, filename=display_name, chunk_count=len(texts),
        status="completed", created_at=datetime.datetime.now(timezone.utc).isoformat(),
        owner_id=EVAL_OWNER_ID,
    )

    print(f"Ingested '{display_name}': {len(texts)} chunks, {len(parent_store)} parent segments "
          f"(CHILD_CHUNK_SIZE env={os.getenv('CHILD_CHUNK_SIZE','<default>')}).")

    # Knowledge Graph Extraction
    try:
        max_segments = int(os.getenv("GRAPH_MAX_SEGMENTS", "15"))
    except (ValueError, TypeError):
        max_segments = 15
    max_segments = max(0, max_segments)

    parent_candidates = list(parent_store.values()) if parent_store else texts
    graph_texts = parent_candidates[:max_segments]

    print(f"Starting graph extraction for {len(graph_texts)} parent segments "
          f"(capped by GRAPH_MAX_SEGMENTS={max_segments}, out of {len(parent_candidates)} total)...")

    sem = asyncio.Semaphore(2)
    processed_count = 0
    error_count = 0

    async def bounded_extract(idx: int, t: str):
        nonlocal processed_count, error_count
        async with sem:
            try:
                print(f"Extracting graph segment {idx + 1}/{len(graph_texts)}...")
                await extract_and_store_graph(t)
                await asyncio.sleep(1)
            except Exception as e:
                error_count += 1
                print(f"Graph extraction warning on segment {idx + 1} (non-fatal): {e}")
            finally:
                processed_count += 1

    if graph_texts:
        try:
            await asyncio.gather(*(bounded_extract(i, t) for i, t in enumerate(graph_texts)), return_exceptions=True)
        except Exception as e:
            print(f"Graph extraction gather error (non-fatal): {e}")

    print(f"Graph extraction complete: {processed_count} graph segments processed "
          f"({error_count} errors, {processed_count - error_count} successful).")

    return len(texts)


if __name__ == "__main__":
    default_doc = os.path.join(TEST_DOCS_DIR, "well_deliverability.pdf")
    ap = argparse.ArgumentParser(description="Ingest document with graph extraction for evaluation.")
    ap.add_argument("path", nargs="?", default=default_doc, help="path to the PDF/DOCX to ingest (default: well_deliverability.pdf)")
    ap.add_argument("--name", default=None, help="display filename to store in metadata")
    args = ap.parse_args()

    name = args.name or os.path.basename(args.path)
    asyncio.run(ingest_with_graph(args.path, name))
