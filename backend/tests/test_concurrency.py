"""
Concurrency tests — verify the sparse store survives parallel multi-user access.

We migrated the sparse store toward multi-worker operation; these tests exercise
the REAL SQLite FTS5 store under concurrent writers and readers from many threads
to catch races, lock contention deadlocks, and data corruption. The store uses
check_same_thread=False + an internal lock, so it must serialize writes correctly
without losing rows or leaking across owners.

No mocks: a real temp SQLite DB is used.
"""

import os
import uuid
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest


@pytest.fixture
def real_store():
    tmpdir = tempfile.mkdtemp(prefix="sbrag_conc_")
    old = os.environ.get("BM25_DATA_DIR")
    os.environ["BM25_DATA_DIR"] = tmpdir

    import app.rag.vectorstore.bm25_store as bm25_mod
    import sqlite3
    store = bm25_mod.BM25Store.__new__(bm25_mod.BM25Store)
    store.conn = sqlite3.connect(os.path.join(tmpdir, "bm25.db"), check_same_thread=False, timeout=30.0)
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
    store._lock = threading.Lock()

    yield store

    try:
        store.conn.close()
    except Exception:
        pass
    if old is None:
        os.environ.pop("BM25_DATA_DIR", None)
    else:
        os.environ["BM25_DATA_DIR"] = old
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestConcurrentWrites:
    def test_parallel_writes_lose_no_rows(self, real_store):
        """20 threads each insert 10 unique chunks; all 200 must persist (no lost writes)."""
        store = real_store
        n_threads = 20
        per_thread = 10

        def writer(tid):
            docs = [
                {
                    "id": f"t{tid}-c{i}",
                    "text": f"thread {tid} chunk {i} about topic alpha beta gamma",
                    "doc_id": f"owner{tid}:doc",
                    "owner_id": f"owner{tid}",
                }
                for i in range(per_thread)
            ]
            store.add_documents(docs)

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = [ex.submit(writer, t) for t in range(n_threads)]
            for f in as_completed(futures):
                f.result()  # re-raise any thread exception

        # Verify total row count in the FTS corpus equals exactly n_threads * per_thread
        cur = store.conn.execute("SELECT COUNT(*) FROM corpus")
        total = cur.fetchone()[0]
        assert total == n_threads * per_thread, f"Lost writes: expected {n_threads*per_thread}, got {total}"

    def test_concurrent_writes_and_searches_no_corruption(self, real_store):
        """Interleave writers and readers; searches must never crash or return foreign owners."""
        store = real_store
        errors = []

        def writer(tid):
            try:
                store.add_documents([{
                    "id": f"w{tid}", "text": f"payments service owner {tid} distributed ledger",
                    "doc_id": f"owner{tid}:doc", "owner_id": f"owner{tid}",
                }])
            except Exception as e:  # pragma: no cover
                errors.append(("write", tid, repr(e)))

        def reader(tid):
            try:
                res = store.search("distributed ledger", top_k=5, owner_id=f"owner{tid}")
                # Every returned chunk (if any) must belong to this owner
                for r in res:
                    assert r["id"] == f"w{tid}", f"owner{tid} saw foreign chunk {r['id']}"
            except Exception as e:
                errors.append(("read", tid, repr(e)))

        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = []
            for t in range(20):
                futures.append(ex.submit(writer, t))
                futures.append(ex.submit(reader, t))
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Concurrency errors detected: {errors[:5]}"

    def test_parallel_metadata_upserts_are_consistent(self, real_store):
        """Concurrent metadata upserts for distinct docs must all land, none lost."""
        store = real_store

        def upsert(tid):
            store.update_document_metadata(
                doc_id=f"owner{tid}:doc", filename=f"f{tid}.pdf",
                chunk_count=tid, status="completed", created_at="t", owner_id=f"owner{tid}",
            )

        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(upsert, range(16)))

        assert store.get_total_docs() == 16, "Concurrent metadata upserts lost rows"
        # Spot-check owner isolation on metadata reads
        for t in range(16):
            docs = store.get_document_metadata(owner_id=f"owner{t}")
            assert len(docs) == 1 and docs[0]["owner_id"] == f"owner{t}"


class TestConcurrentCacheGeneration:
    """The per-owner cache generation counter must be race-safe under parallel bumps."""

    @pytest.mark.asyncio
    async def test_parallel_bumps_are_monotonic(self):
        import asyncio
        import app.core.cache as cache_mod
        await cache_mod.init_cache()

        owner = f"race-{uuid.uuid4().hex[:8]}"
        start = await cache_mod.get_cache_generation(owner)

        # Fire many concurrent bumps
        await asyncio.gather(*(cache_mod.bump_cache_generation(owner) for _ in range(50)))

        end = await cache_mod.get_cache_generation(owner)
        assert end == start + 50, f"Cache generation lost increments under concurrency: {start} -> {end}"
