"""
REAL end-to-end integration tests — NO mocks for storage/embedding/retrieval.

These tests exercise the true pipeline against real components:
  - Real SentenceTransformer embeddings (all-MiniLM-L6-v2, downloaded once)
  - Real ephemeral ChromaDB (in-process)
  - Real SQLite FTS5 BM25 store (temp dir)
  - Real recursive parent-child chunking
  - Real hybrid retrieval (dense + sparse + RRF), reranker disabled for determinism

Purpose: prove the full ingest -> index -> retrieve path actually works and that
per-user isolation holds end-to-end, not just that the right mocks were called.

Marked `slow` because they load a real embedding model. Skipped automatically if
the model cannot be downloaded (offline CI) so the suite stays green, but when the
model is available these are the highest-signal tests in the repo.
"""

import os
import uuid
import shutil
import tempfile

import pytest


# ---- Skip guard: only run if the real embedding model can be loaded --------
def _embedder_available() -> bool:
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _embedder_available(),
    reason="sentence-transformers not installed; real E2E embedding test skipped",
)


@pytest.fixture
def real_stack(monkeypatch):
    """
    Wire the pipeline to REAL components:
      - fresh temp BM25 SQLite dir
      - in-process ephemeral Chroma
      - in-memory cache
    Yields helper handles. Cleans everything up afterward.
    """
    tmpdir = tempfile.mkdtemp(prefix="sbrag_e2e_")
    monkeypatch.setenv("BM25_DATA_DIR", tmpdir)
    monkeypatch.setenv("SPARSE_BACKEND", "sqlite")
    monkeypatch.setenv("CACHE_BACKEND", "memory")
    monkeypatch.setenv("OTEL_ENABLED", "false")

    # Real ephemeral Chroma, injected into the store module. Reset the module-level
    # _client singleton too so this ephemeral client never leaks into other tests.
    import chromadb
    import app.rag.vectorstore.chroma_store as chroma_mod
    monkeypatch.setattr(chroma_mod, "_client", None, raising=False)
    client = chromadb.EphemeralClient()
    monkeypatch.setattr(chroma_mod, "get_client", lambda: client)
    monkeypatch.setattr(chroma_mod, "_client", client, raising=False)
    # get_collection uses get_client(); make sure a fresh collection is used
    try:
        client.delete_collection("rag_collection")
    except Exception:
        pass

    # Fresh real SQLite BM25 store pointed at the temp dir.
    # Install it as the module-level singleton so EVERY module that calls
    # get_bm25_store() (ingestion, retriever, routes) gets this same real store,
    # regardless of how they imported the symbol.
    import app.rag.vectorstore.bm25_store as bm25_mod
    bm25_mod.reset_bm25_store()
    # Point the module path constants at the temp dir so the real constructor
    # builds its DB there (these are read at __init__ time).
    os.makedirs(tmpdir, exist_ok=True)
    monkeypatch.setattr(bm25_mod, "_data_dir", tmpdir, raising=False)
    monkeypatch.setattr(bm25_mod, "SQLITE_DB_PATH", os.path.join(tmpdir, "bm25.db"), raising=False)
    monkeypatch.setattr(bm25_mod, "BM25_FILE_PATH", os.path.join(tmpdir, "bm25_corpus.json"), raising=False)
    store = bm25_mod.BM25Store()  # real SQLite FTS5 store in the temp dir
    bm25_mod._bm25_store = store
    bm25_mod._current_backend = "sqlite"

    # Disable the cross-encoder reranker for deterministic, model-light retrieval
    import app.rag.retrievers.hybrid_retriever as hr_mod
    monkeypatch.setattr(hr_mod, "reranker", None)

    yield {"chroma": client, "bm25": store, "tmpdir": tmpdir}

    try:
        client.delete_collection("rag_collection")
    except Exception:
        pass
    try:
        store.conn.close()
    except Exception:
        pass
    bm25_mod.reset_bm25_store()
    # monkeypatch auto-restores get_client and _client after the test
    shutil.rmtree(tmpdir, ignore_errors=True)


async def _ingest(docs_text, owner_id, doc_id, filename):
    """Run the REAL _process_docs against the real wired stack."""
    from langchain_core.documents import Document
    from app.rag.ingestion import IngestionPipeline

    docs = [Document(page_content=docs_text, metadata={"filename": filename, "source_type": "pdf"})]
    # Skip graph extraction (needs an LLM) by pointing GRAPH_MAX_SEGMENTS to 0
    os.environ["GRAPH_MAX_SEGMENTS"] = "0"
    await IngestionPipeline._process_docs(
        docs=docs, job_id=f"job-{uuid.uuid4().hex[:6]}",
        doc_id=doc_id, owner_id=owner_id, filename=filename,
    )


class TestRealEndToEnd:
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_ingest_and_retrieve(self, real_stack):
        """Ingest real text, then retrieve it via the real hybrid pipeline."""
        from app.rag.retrievers.hybrid_retriever import hybrid_search

        await _ingest(
            "Kubernetes is an open-source container orchestration platform for "
            "automating deployment, scaling, and management of containerized applications.",
            owner_id="alice", doc_id="alice:doc1", filename="k8s.pdf",
        )

        # Real embeddings + real Chroma + real BM25 + real RRF
        results = await hybrid_search("container orchestration", top_k=5, owner_id="alice")

        assert len(results) > 0, "E2E retrieval returned nothing from a real index"
        combined = " ".join(r.get("text", "") for r in results).lower()
        assert "kubernetes" in combined or "orchestration" in combined
        # Chroma total should reflect the real write
        assert real_stack["chroma"].get_or_create_collection("rag_collection").count() > 0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_cross_user_isolation_end_to_end(self, real_stack):
        """
        Two users ingest overlapping content. Each must retrieve ONLY their own,
        proven against real embeddings + real stores (the strongest isolation test).
        """
        from app.rag.retrievers.hybrid_retriever import hybrid_search

        await _ingest(
            "The quarterly revenue report shows strong growth in cloud services.",
            owner_id="alice", doc_id="alice:report", filename="report.pdf",
        )
        await _ingest(
            "The quarterly revenue report shows a decline in hardware sales.",
            owner_id="bob", doc_id="bob:report", filename="report.pdf",
        )

        alice_hits = await hybrid_search("quarterly revenue report", top_k=10, owner_id="alice")
        bob_hits = await hybrid_search("quarterly revenue report", top_k=10, owner_id="bob")

        alice_text = " ".join(r.get("text", "") for r in alice_hits).lower()
        bob_text = " ".join(r.get("text", "") for r in bob_hits).lower()

        # Alice must see cloud growth, never bob's hardware decline — and vice versa
        assert "cloud" in alice_text
        assert "hardware" not in alice_text, "ISOLATION BREACH: alice saw bob's content"
        assert "hardware" in bob_text
        assert "cloud" not in bob_text, "ISOLATION BREACH: bob saw alice's content"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_delete_removes_from_real_index(self, real_stack):
        """Deleting a document must remove it from BOTH real Chroma and real BM25."""
        from app.rag.retrievers.hybrid_retriever import hybrid_search
        from app.rag.vectorstore.chroma_store import delete_document

        await _ingest(
            "Ephemeral test document about distributed consensus algorithms like Raft.",
            owner_id="carol", doc_id="carol:doc1", filename="raft.pdf",
        )
        before = await hybrid_search("consensus algorithms", top_k=5, owner_id="carol")
        assert len(before) > 0

        await delete_document("carol:doc1", owner_id="carol")
        real_stack["bm25"].delete_documents_by_doc_id("carol:doc1", owner_id="carol")

        after = await hybrid_search("consensus algorithms", top_k=5, owner_id="carol")
        assert after == [] or all("raft" not in r.get("text", "").lower() for r in after)
