import os
import uuid
from concurrent.futures import ThreadPoolExecutor
import pytest

from app.rag.vectorstore.bm25_store import (
    BM25Store,
    SQLiteBM25Store,
    PostgresBM25Store,
    get_bm25_store,
    reset_bm25_store,
)

DATABASE_URL = os.getenv("DATABASE_URL")
requires_postgres = pytest.mark.skipif(
    not DATABASE_URL,
    reason="DATABASE_URL environment variable is not set (PostgreSQL tests skipped)",
)


def test_factory_sqlite_default(monkeypatch):
    """Test get_bm25_store returns SQLite store by default and when SPARSE_BACKEND=sqlite."""
    reset_bm25_store()
    monkeypatch.delenv("SPARSE_BACKEND", raising=False)
    store = get_bm25_store()
    assert isinstance(store, BM25Store)
    assert isinstance(store, SQLiteBM25Store)

    reset_bm25_store()
    monkeypatch.setenv("SPARSE_BACKEND", "sqlite")
    store = get_bm25_store()
    assert isinstance(store, BM25Store)


def test_factory_postgres_missing_database_url(monkeypatch):
    """Test that selecting postgres backend without DATABASE_URL raises ValueError."""
    reset_bm25_store()
    monkeypatch.setenv("SPARSE_BACKEND", "postgres")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValueError, match="DATABASE_URL"):
        get_bm25_store()

    reset_bm25_store()


@requires_postgres
def test_factory_postgres_selected(monkeypatch):
    """Test get_bm25_store returns PostgresBM25Store when SPARSE_BACKEND=postgres."""
    reset_bm25_store()
    monkeypatch.setenv("SPARSE_BACKEND", "postgres")
    store = get_bm25_store()
    assert isinstance(store, PostgresBM25Store)
    reset_bm25_store()


def test_sqlite_store_remains_intact():
    """Verify the existing SQLite store remains fully working with its public interface."""
    store = BM25Store()
    doc_id = f"sqlite_test_{uuid.uuid4().hex[:8]}"
    chunk_id = f"{doc_id}_c1"

    try:
        # 1. Add document
        store.add_documents([
            {"id": chunk_id, "doc_id": doc_id, "text": "SQLite full text search using fts5 index.", "owner_id": "sqlite_owner"}
        ])

        # 2. Search
        results = store.search("fts5", top_k=3)
        assert len(results) > 0
        assert results[0]["id"] == chunk_id
        assert "text" in results[0]
        assert "score" in results[0]
        assert isinstance(results[0]["score"], float)

        # 2b. Search with matching owner_id
        owner_results = store.search("fts5", top_k=3, owner_id="sqlite_owner")
        assert len(owner_results) > 0
        assert owner_results[0]["id"] == chunk_id

        # 2c. Search with non-matching owner_id
        other_results = store.search("fts5", top_k=3, owner_id="different_owner")
        assert other_results == []

        # 3. Empty query
        assert store.search("") == []
        assert store.search("   ") == []

        # 4. Metadata and owner filter
        store.update_document_metadata(
            doc_id=doc_id,
            filename="sqlite.txt",
            chunk_count=1,
            status="indexed",
            created_at="2026-01-01T00:00:00",
            owner_id="sqlite_owner"
        )
        meta = store.get_document(doc_id)
        assert meta is not None
        assert meta["filename"] == "sqlite.txt"
        assert meta["owner_id"] == "sqlite_owner"

        filtered = store.get_document_metadata(owner_id="sqlite_owner")
        assert any(d["doc_id"] == doc_id for d in filtered)

        # 5. COALESCE owner_id update
        store.update_document_metadata(
            doc_id=doc_id,
            filename="sqlite_updated.txt",
            chunk_count=1,
            status="reindexed",
            created_at="2026-01-02T00:00:00",
            owner_id=None
        )
        meta_updated = store.get_document(doc_id)
        assert meta_updated["filename"] == "sqlite_updated.txt"
        assert meta_updated["owner_id"] == "sqlite_owner"

        # 6. Delete
        store.delete_documents_by_doc_id(doc_id)
        assert store.get_document(doc_id) is None
    finally:
        store.delete_documents_by_doc_id(doc_id)


def test_sqlite_store_owner_isolation_search_and_delete():
    """Verify SQLite store isolates searches and deletes across different owners."""
    store = BM25Store()
    user_a_doc = f"sqlite_ua_{uuid.uuid4().hex[:8]}"
    user_b_doc = f"sqlite_ub_{uuid.uuid4().hex[:8]}"
    c_a = f"{user_a_doc}_c1"
    c_b = f"{user_b_doc}_c1"

    try:
        store.add_documents([
            {"id": c_a, "doc_id": user_a_doc, "text": "Kubernetes orchestration and container deployment.", "owner_id": "user_a"},
            {"id": c_b, "doc_id": user_b_doc, "text": "Kubernetes networking and service mesh deployment.", "owner_id": "user_b"},
        ])
        store.update_document_metadata(user_a_doc, "k8s_a.pdf", 1, "completed", "2026-01-01", "user_a")
        store.update_document_metadata(user_b_doc, "k8s_b.pdf", 1, "completed", "2026-01-01", "user_b")

        # Search with owner_id filter
        res_a = store.search("Kubernetes", top_k=5, owner_id="user_a")
        assert len(res_a) == 1
        assert res_a[0]["id"] == c_a

        res_b = store.search("Kubernetes", top_k=5, owner_id="user_b")
        assert len(res_b) == 1
        assert res_b[0]["id"] == c_b

        # Unfiltered search returns both
        res_all = store.search("Kubernetes", top_k=5)
        assert len(res_all) == 2

        # Owner-scoped delete: attempting to delete user_a's doc as user_b should not delete it
        store.delete_documents_by_doc_id(user_a_doc, owner_id="user_b")
        assert store.get_document(user_a_doc) is not None
        assert len(store.search("Kubernetes", top_k=5, owner_id="user_a")) == 1

        # Deleting user_a's doc as user_a succeeds
        store.delete_documents_by_doc_id(user_a_doc, owner_id="user_a")
        assert store.get_document(user_a_doc) is None
        assert len(store.search("Kubernetes", top_k=5, owner_id="user_a")) == 0

        # user_b's doc remains untouched
        assert store.get_document(user_b_doc) is not None
        assert len(store.search("Kubernetes", top_k=5, owner_id="user_b")) == 1
    finally:
        store.delete_documents_by_doc_id(user_a_doc)
        store.delete_documents_by_doc_id(user_b_doc)


@pytest.fixture
def pg_store():
    """Fixture providing a PostgresBM25Store instance with automatic test doc cleanup."""
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")

    store = PostgresBM25Store(database_url=DATABASE_URL)
    created_doc_ids = []

    class TestStoreWrapper:
        def __init__(self, inner):
            self.inner = inner

        def track(self, doc_id):
            created_doc_ids.append(doc_id)
            return doc_id

        def __getattr__(self, name):
            return getattr(self.inner, name)

    wrapper = TestStoreWrapper(store)
    yield wrapper

    for doc_id in created_doc_ids:
        try:
            store.delete_documents_by_doc_id(doc_id)
        except Exception:
            pass


@requires_postgres
def test_postgres_store_add_and_search(pg_store):
    """Test PostgresBM25Store add_documents and search functionality."""
    doc_id = pg_store.track(f"pg_test_{uuid.uuid4().hex[:8]}")
    c1 = f"{doc_id}_c1"
    c2 = f"{doc_id}_c2"

    pg_store.add_documents([
        {"id": c1, "doc_id": doc_id, "text": "PostgreSQL full-text-search with tsvector and GIN indexes."},
        {"id": c2, "doc_id": doc_id, "text": "Unrelated culinary recipes about chocolate cookies and milk."},
    ])

    # 1. Search for matching keywords
    results = pg_store.search("tsvector indexes", top_k=5)
    assert len(results) > 0
    matched_ids = [r["id"] for r in results]
    assert c1 in matched_ids
    top_match = next(r for r in results if r["id"] == c1)
    assert "tsvector" in top_match["text"]
    assert isinstance(top_match["score"], float)
    assert top_match["score"] > 0

    # 2. Empty and whitespace queries must return []
    assert pg_store.search("") == []
    assert pg_store.search("   ") == []
    assert pg_store.search("", top_k=0) == []

    # 3. Non-matching query must return []
    assert [r for r in pg_store.search("nonexistentqueryxylophone999") if r["id"] in (c1, c2)] == []


@requires_postgres
def test_postgres_store_delete_documents(pg_store):
    """Test PostgresBM25Store delete_documents_by_doc_id clears corpus and metadata."""
    doc_id = pg_store.track(f"pg_test_del_{uuid.uuid4().hex[:8]}")
    chunk_id = f"{doc_id}_chunk"

    pg_store.add_documents([
        {"id": chunk_id, "doc_id": doc_id, "text": "Quantum entanglement in photon physics experiments."}
    ])
    pg_store.update_document_metadata(
        doc_id=doc_id,
        filename="quantum.pdf",
        chunk_count=1,
        status="indexed",
        created_at="2026-01-01T00:00:00",
        owner_id="owner_quantum"
    )

    # Verify present
    assert pg_store.get_document(doc_id) is not None
    search_res = pg_store.search("quantum photon", top_k=5)
    assert any(r["id"] == chunk_id for r in search_res)

    # Delete
    pg_store.delete_documents_by_doc_id(doc_id)

    # Verify removed from corpus and document_metadata
    assert pg_store.get_document(doc_id) is None
    search_after = pg_store.search("quantum photon", top_k=5)
    assert not any(r["id"] == chunk_id for r in search_after)


@requires_postgres
def test_postgres_store_owner_isolation_search_and_delete(pg_store):
    """Test that PostgresBM25Store isolates searches and deletes by owner_id."""
    doc_a = pg_store.track(f"pg_iso_a_{uuid.uuid4().hex[:8]}")
    doc_b = pg_store.track(f"pg_iso_b_{uuid.uuid4().hex[:8]}")
    c_a = f"{doc_a}_c1"
    c_b = f"{doc_b}_c1"

    pg_store.add_documents([
        {"id": c_a, "doc_id": doc_a, "text": "Astrophysics and relativistic cosmology theories.", "owner_id": "user_astro"},
        {"id": c_b, "doc_id": doc_b, "text": "Astrophysics and observational astronomy instrumentation.", "owner_id": "user_obs"},
    ])
    pg_store.update_document_metadata(doc_a, "cosmo.pdf", 1, "completed", "2026-01-01", "user_astro")
    pg_store.update_document_metadata(doc_b, "obs.pdf", 1, "completed", "2026-01-01", "user_obs")

    # Search with owner_id filter
    res_a = pg_store.search("Astrophysics", top_k=5, owner_id="user_astro")
    assert any(r["id"] == c_a for r in res_a)
    assert not any(r["id"] == c_b for r in res_a)

    res_b = pg_store.search("Astrophysics", top_k=5, owner_id="user_obs")
    assert any(r["id"] == c_b for r in res_b)
    assert not any(r["id"] == c_a for r in res_b)

    # Search without owner_id returns both
    res_all = pg_store.search("Astrophysics", top_k=5)
    assert any(r["id"] == c_a for r in res_all)
    assert any(r["id"] == c_b for r in res_all)

    # Owner-scoped delete: trying to delete doc_a with wrong owner should do nothing
    pg_store.delete_documents_by_doc_id(doc_a, owner_id="user_obs")
    assert pg_store.get_document(doc_a) is not None
    assert any(r["id"] == c_a for r in pg_store.search("Astrophysics", top_k=5, owner_id="user_astro"))

    # Deleting with correct owner should succeed
    pg_store.delete_documents_by_doc_id(doc_a, owner_id="user_astro")
    assert pg_store.get_document(doc_a) is None
    assert not any(r["id"] == c_a for r in pg_store.search("Astrophysics", top_k=5, owner_id="user_astro"))
    # doc_b still remains
    assert pg_store.get_document(doc_b) is not None


@requires_postgres
def test_postgres_store_metadata_and_owner_filter(pg_store):
    """Test metadata upsert, filtering by owner_id and doc_id, COALESCE logic, and get_total_docs."""
    alice_doc = pg_store.track(f"pg_alice_{uuid.uuid4().hex[:8]}")
    bob_doc = pg_store.track(f"pg_bob_{uuid.uuid4().hex[:8]}")

    pg_store.update_document_metadata(
        doc_id=alice_doc,
        filename="alice_report.pdf",
        chunk_count=10,
        status="indexed",
        created_at="2026-03-01T10:00:00",
        owner_id="alice_123"
    )
    pg_store.update_document_metadata(
        doc_id=bob_doc,
        filename="bob_notes.txt",
        chunk_count=3,
        status="indexed",
        created_at="2026-03-01T11:00:00",
        owner_id="bob_456"
    )

    # Filter by owner_id alice
    alice_docs = pg_store.get_document_metadata(owner_id="alice_123")
    alice_ids = [d["doc_id"] for d in alice_docs]
    assert alice_doc in alice_ids
    assert bob_doc not in alice_ids

    # Filter by owner_id bob
    bob_docs = pg_store.get_document_metadata(owner_id="bob_456")
    bob_ids = [d["doc_id"] for d in bob_docs]
    assert bob_doc in bob_ids
    assert alice_doc not in bob_ids

    # Filter by doc_id
    doc_res = pg_store.get_document_metadata(doc_id=alice_doc)
    assert len(doc_res) == 1
    assert doc_res[0]["filename"] == "alice_report.pdf"

    # get_document
    single_doc = pg_store.get_document(alice_doc)
    assert single_doc is not None
    assert single_doc["doc_id"] == alice_doc
    assert single_doc["chunk_count"] == 10
    assert single_doc["owner_id"] == "alice_123"

    # Non-existent doc
    assert pg_store.get_document("nonexistent_doc_id_9999") is None

    # COALESCE update: preserve owner_id when new owner_id is None
    pg_store.update_document_metadata(
        doc_id=alice_doc,
        filename="alice_report_v2.pdf",
        chunk_count=15,
        status="updated",
        created_at="2026-03-02T10:00:00",
        owner_id=None
    )
    alice_after_update = pg_store.get_document(alice_doc)
    assert alice_after_update["filename"] == "alice_report_v2.pdf"
    assert alice_after_update["chunk_count"] == 15
    assert alice_after_update["status"] == "updated"
    assert alice_after_update["owner_id"] == "alice_123"

    # get_total_docs
    total = pg_store.get_total_docs()
    assert isinstance(total, int)
    assert total >= 2


@requires_postgres
def test_postgres_store_upsert_chunk(pg_store):
    """Test upserting existing chunk_id updates text without duplicate rows."""
    doc_id = pg_store.track(f"pg_upsert_{uuid.uuid4().hex[:8]}")
    chunk_id = f"{doc_id}_chunk1"

    # 1. First insert
    pg_store.add_documents([
        {"id": chunk_id, "doc_id": doc_id, "text": "Initial text about astronomy and planetary orbits."}
    ])
    res1 = pg_store.search("astronomy", top_k=5)
    assert any(r["id"] == chunk_id for r in res1)

    # 2. Upsert same chunk_id with new content
    pg_store.add_documents([
        {"id": chunk_id, "doc_id": doc_id, "text": "Replaced content describing biotechnology and genetics."}
    ])

    # Old text should no longer match this chunk
    res_old = pg_store.search("astronomy", top_k=5)
    assert not any(r["id"] == chunk_id for r in res_old)

    # New text should match
    res_new = pg_store.search("biotechnology", top_k=5)
    matched = [r for r in res_new if r["id"] == chunk_id]
    assert len(matched) == 1
    assert "biotechnology" in matched[0]["text"]


@requires_postgres
def test_postgres_store_multi_threaded_reads(pg_store):
    """Verify thread-safety when multiple concurrent threads execute queries."""
    doc_id = pg_store.track(f"pg_concur_{uuid.uuid4().hex[:8]}")
    pg_store.add_documents([
        {"id": f"{doc_id}_c1", "doc_id": doc_id, "text": "Concurrent distributed database performance benchmarking."}
    ])

    def do_search(i):
        return pg_store.search("concurrent distributed database", top_k=5)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(do_search, i) for i in range(16)]
        for f in futures:
            results = f.result()
            assert any(r["id"] == f"{doc_id}_c1" for r in results)
