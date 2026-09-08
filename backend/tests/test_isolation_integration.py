"""
REAL integration tests for per-user data isolation — NO mocks for the sparse store.

These tests exercise the actual SQLite FTS5 BM25 store to prove that the
multi-user isolation fixes (owner-scoped search + owner-scoped delete +
per-owner unique doc_id) actually work end-to-end, not just that the right
arguments were passed to a mock.

Why this file exists: the existing unit tests mock the BM25 store, so they
verify wiring but not real database behaviour. These tests use a temporary,
real SQLite database.
"""

import os
import tempfile
import shutil

import pytest


@pytest.fixture
def real_bm25_store():
    """A real SQLite BM25 store backed by a fresh temp directory."""
    tmpdir = tempfile.mkdtemp(prefix="sbrag_iso_test_")
    old_env = os.environ.get("BM25_DATA_DIR")
    old_backend = os.environ.get("SPARSE_BACKEND")
    os.environ["BM25_DATA_DIR"] = tmpdir
    os.environ["SPARSE_BACKEND"] = "sqlite"

    # Use a fresh SQLite store instance pointed at the temp dir WITHOUT reloading
    # the module (reloading pollutes other test files that hold module references).
    import app.rag.vectorstore.bm25_store as bm25_mod
    # BM25_DATA_DIR is read at module import for path constants; construct the store
    # against the temp DB path directly to stay isolated.
    import sqlite3
    store = bm25_mod.BM25Store.__new__(bm25_mod.BM25Store)
    os.makedirs(tmpdir, exist_ok=True)
    store.conn = sqlite3.connect(os.path.join(tmpdir, "bm25.db"), check_same_thread=False)
    store.conn.execute("PRAGMA journal_mode=WAL;")
    store.conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS corpus USING fts5(id UNINDEXED, text, doc_id UNINDEXED, owner_id UNINDEXED);"
    )
    store.conn.execute(
        "CREATE TABLE IF NOT EXISTS corpus_mapping (rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, doc_id TEXT);"
    )
    store.conn.execute(
        "CREATE TABLE IF NOT EXISTS document_metadata (doc_id TEXT PRIMARY KEY, filename TEXT, chunk_count INTEGER, status TEXT, created_at TEXT, owner_id TEXT);"
    )
    store.conn.commit()
    import threading
    store._lock = threading.Lock()

    yield store

    # Cleanup
    try:
        if hasattr(store, "conn"):
            store.conn.close()
    except Exception:
        pass
    bm25_mod.reset_bm25_store()
    if old_env is None:
        os.environ.pop("BM25_DATA_DIR", None)
    else:
        os.environ["BM25_DATA_DIR"] = old_env
    if old_backend is None:
        os.environ.pop("SPARSE_BACKEND", None)
    else:
        os.environ["SPARSE_BACKEND"] = old_backend
    shutil.rmtree(tmpdir, ignore_errors=True)


def _add_doc(store, chunk_id, text, doc_id, owner_id):
    store.add_documents([{"id": chunk_id, "text": text, "doc_id": doc_id, "owner_id": owner_id}])


class TestRealSparseIsolation:
    def test_search_only_returns_owner_chunks(self, real_bm25_store):
        """User A's search must never return User B's chunks."""
        store = real_bm25_store
        # Both users have a doc mentioning 'kubernetes'
        _add_doc(store, "a1", "kubernetes orchestration guide for user A", "userA:doc1", "userA")
        _add_doc(store, "b1", "kubernetes orchestration guide for user B", "userB:doc1", "userB")

        a_results = store.search("kubernetes", top_k=10, owner_id="userA")
        b_results = store.search("kubernetes", top_k=10, owner_id="userB")

        a_ids = {r["id"] for r in a_results}
        b_ids = {r["id"] for r in b_results}

        assert a_ids == {"a1"}, f"User A leaked other chunks: {a_ids}"
        assert b_ids == {"b1"}, f"User B leaked other chunks: {b_ids}"

    def test_delete_is_owner_scoped_with_same_docname(self, real_bm25_store):
        """
        FLAW 1 regression: two users upload docs; deleting one owner's doc
        must NOT affect the other owner, even though they overlap in content.
        """
        store = real_bm25_store
        _add_doc(store, "a1", "quarterly financial report content", "userA:reportuuid", "userA")
        _add_doc(store, "b1", "quarterly financial report content", "userB:reportuuid", "userB")
        store.update_document_metadata("userA:reportuuid", "report.pdf", 1, "completed", "t", owner_id="userA")
        store.update_document_metadata("userB:reportuuid", "report.pdf", 1, "completed", "t", owner_id="userB")

        # User A deletes their doc (owner-scoped)
        store.delete_documents_by_doc_id("userA:reportuuid", owner_id="userA")

        # User A should now have nothing; User B must still have their doc + chunk
        a_docs = store.get_document_metadata(owner_id="userA")
        b_docs = store.get_document_metadata(owner_id="userB")
        assert a_docs == [], f"User A doc not deleted: {a_docs}"
        assert len(b_docs) == 1, f"User B doc wrongly affected: {b_docs}"

        b_results = store.search("financial report", top_k=10, owner_id="userB")
        assert {r["id"] for r in b_results} == {"b1"}, "User B chunk wrongly deleted"

    def test_list_metadata_is_owner_scoped(self, real_bm25_store):
        store = real_bm25_store
        store.update_document_metadata("userA:d1", "a.pdf", 2, "completed", "t", owner_id="userA")
        store.update_document_metadata("userB:d1", "b.pdf", 3, "completed", "t", owner_id="userB")

        a_docs = store.get_document_metadata(owner_id="userA")
        assert len(a_docs) == 1 and a_docs[0]["owner_id"] == "userA"
        assert a_docs[0]["doc_id"] == "userA:d1"

    def test_search_without_owner_is_global_backcompat(self, real_bm25_store):
        """owner_id=None must preserve original global behaviour."""
        store = real_bm25_store
        _add_doc(store, "a1", "distributed systems raft consensus", "userA:d1", "userA")
        _add_doc(store, "b1", "distributed systems paxos consensus", "userB:d1", "userB")

        results = store.search("consensus", top_k=10, owner_id=None)
        assert {r["id"] for r in results} == {"a1", "b1"}, "Global search should see all owners"


class TestRealCacheGeneration:
    """FLAW 8 regression: cache generation bump must be per-owner."""

    @pytest.mark.asyncio
    async def test_cache_generation_is_per_owner(self):
        import app.core.cache as cache_mod
        # Use unique owner ids so this test is independent of any shared cache state
        import uuid
        owner_a = f"userA-{uuid.uuid4().hex[:8]}"
        owner_b = f"userB-{uuid.uuid4().hex[:8]}"

        gen_a0 = await cache_mod.get_cache_generation(owner_a)
        await cache_mod.bump_cache_generation(owner_a)
        gen_a1 = await cache_mod.get_cache_generation(owner_a)
        gen_b = await cache_mod.get_cache_generation(owner_b)

        assert gen_a1 == gen_a0 + 1, "User A generation did not bump"
        assert gen_b == 0, "User B generation wrongly affected by User A bump"
