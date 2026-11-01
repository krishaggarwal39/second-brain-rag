"""
Owner-scoped ingestion of a SINGLE real PDF/DOCX document for evaluation.

Usage:
  python ingest_doc.py <path-to-file> [--name display_name.pdf]
  python ingest_doc.py test_documents/well_deliverability.pdf

Ingests under EVAL_OWNER_ID using the same safe lower-level path as the real
pipeline's _process_docs (loader -> chunk -> embed -> cache parents -> chroma
-> bm25), setting owner_id/parent_id/filename, and SKIPPING graph extraction.
Honors CHILD_CHUNK_SIZE / CHILD_CHUNK_OVERLAP env overrides via the chunker.

The 'filename' stored in metadata (and matched by gold_qa_well.json's source_doc)
defaults to the file's basename; override with --name.
"""

import os
import sys
import argparse
import asyncio

import eval_common  # noqa: F401
from eval_common import EVAL_OWNER_ID


async def ingest(path: str, display_name: str) -> int:
    from app.rag.loaders.pdf_loader import load_pdf
    from app.rag.loaders.docx_loader import load_docx
    from app.rag.chunkers.recursive_chunker import chunk_documents
    from app.rag.embeddings.local_embedder import embed_documents
    from app.rag.vectorstore.chroma_store import add_documents
    from app.rag.vectorstore.bm25_store import get_bm25_store
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

    embeddings = await embed_documents(texts)
    if not embeddings or len(embeddings) != len(texts):
        raise ProcessingError("embedding generation failed")

    for pid, ptext in parent_store.items():
        await set_cache(f"parent:{pid}", ptext, ttl=None)

    await add_documents(texts, embeddings, metadatas)

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
    return len(texts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to the PDF/DOCX to ingest")
    ap.add_argument("--name", default=None, help="display filename to store in metadata")
    args = ap.parse_args()
    name = args.name or os.path.basename(args.path)
    asyncio.run(ingest(args.path, name))
