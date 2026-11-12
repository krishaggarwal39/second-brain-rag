"""Unit tests for parent chunks durability, read-through cache, and invalidation (R-12)."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.documents import Document

from app.rag.vectorstore.bm25_store import BM25Store
from app.rag.retrievers.hybrid_retriever import hybrid_search
from app.core.cache import get_cache, delete_cache, get_metric


class TestParentDurabilityR12:
    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        db_dir = tmp_path / "bm25_data"
        monkeypatch.setenv("BM25_DATA_DIR", str(db_dir))
        return BM25Store()

    def test_sqlite_parent_crud(self, store):
        """Test durable parent chunk operations in SQLite parents table."""
        parents = {
            "p-101": "Parent context chunk 101 text.",
            "p-102": "Parent context chunk 102 text.",
        }
        store.add_parents(parents, doc_id="doc-99", owner_id="user-1")

        # Get parent by ID
        assert store.get_parent("p-101") == "Parent context chunk 101 text."
        assert store.get_parent("p-102") == "Parent context chunk 102 text."
        assert store.get_parent("nonexistent") is None

        # Get parent IDs by doc_id
        parent_ids = store.get_parents_by_doc_id("doc-99", owner_id="user-1")
        assert set(parent_ids) == {"p-101", "p-102"}

        # Delete parents for doc_id
        store.delete_parents_by_doc_id("doc-99", owner_id="user-1")
        assert store.get_parent("p-101") is None
        assert store.get_parents_by_doc_id("doc-99") == []

    @pytest.mark.asyncio
    async def test_hybrid_search_read_through_cache_fetches_from_db_and_populates_cache(self):
        """When parent chunk is missing in Redis cache, it is read through from DB and cached."""
        mock_dense = [
            {
                "id": "c1",
                "text": "child text",
                "metadata": {
                    "doc_id": "doc-1",
                    "parent_id": "parent-fallback-1",
                    "owner_id": "user-1",
                },
            }
        ]

        # Ensure parent-fallback-1 is NOT in cache
        await delete_cache("parent:parent-fallback-1")

        with patch("app.rag.retrievers.hybrid_retriever.embed_documents", new_callable=AsyncMock, return_value=[[0.1] * 384]), \
             patch("app.rag.retrievers.hybrid_retriever.chroma_query", new_callable=AsyncMock, return_value=mock_dense), \
             patch("app.rag.retrievers.hybrid_retriever.get_bm25_store") as mock_bm25:

            mock_store = MagicMock()
            mock_store.search.return_value = []
            mock_store.get_parent.return_value = "Durable Parent Text from DB"
            mock_bm25.return_value = mock_store

            results = await hybrid_search("test query", top_k=1, owner_id="user-1")

            assert len(results) >= 1
            assert results[0].get("parent_content") == "Durable Parent Text from DB"
            mock_store.get_parent.assert_called_once_with("parent-fallback-1")

            # Verify read-through populated the cache for subsequent calls
            cached = await get_cache("parent:parent-fallback-1")
            assert cached == "Durable Parent Text from DB"

    @pytest.mark.asyncio
    async def test_hybrid_search_logs_warning_and_increments_parent_missing_metric(self):
        """When child has parent_id but parent text is missing in both cache and DB, record parent_missing."""
        mock_dense = [
            {
                "id": "c_missing",
                "text": "child without parent",
                "metadata": {
                    "doc_id": "doc-2",
                    "parent_id": "parent-nowhere",
                    "owner_id": "user-1",
                },
            }
        ]

        await delete_cache("parent:parent-nowhere")

        with patch("app.rag.retrievers.hybrid_retriever.embed_documents", new_callable=AsyncMock, return_value=[[0.1] * 384]), \
             patch("app.rag.retrievers.hybrid_retriever.chroma_query", new_callable=AsyncMock, return_value=mock_dense), \
             patch("app.rag.retrievers.hybrid_retriever.get_bm25_store") as mock_bm25, \
             patch("app.rag.retrievers.hybrid_retriever.logger.warning") as mock_warn:

            mock_store = MagicMock()
            mock_store.search.return_value = []
            mock_store.get_parent.return_value = None  # Missing in DB too
            mock_bm25.return_value = mock_store

            initial_metric = await get_metric("parent_missing")

            await hybrid_search("test query", top_k=1, owner_id="user-1")

            # Warning logged
            mock_warn.assert_any_call(
                "parent_missing", parent_id="parent-nowhere", doc_id="doc-2"
            )

            # Metric counter incremented
            after_metric = await get_metric("parent_missing")
            assert after_metric == initial_metric + 1.0

    @pytest.mark.asyncio
    async def test_ingestion_failure_after_data_modified_bumps_cache_generation(self, tmp_path):
        """R-12: If ingestion fails after writing data, bump cache generation."""
        pdf_path = str(tmp_path / "mod_fail.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 dummy")

        mock_docs = [Document(page_content="some text", metadata={"filename": "mod_fail.pdf"})]

        with patch("app.rag.ingestion.load_pdf", new_callable=AsyncMock, return_value=mock_docs), \
             patch("app.rag.ingestion.chunk_documents", return_value=(mock_docs, {"p": "parent text"})), \
             patch("app.rag.ingestion.embed_documents", new_callable=AsyncMock, return_value=[[0.1] * 384]), \
             patch("app.rag.ingestion.add_documents", new_callable=AsyncMock, side_effect=RuntimeError("Chroma crashed during insert")), \
             patch("app.rag.ingestion.get_bm25_store") as mock_bm25, \
             patch("app.rag.ingestion.bump_cache_generation", new_callable=AsyncMock) as mock_bump:

            mock_store = MagicMock()
            mock_bm25.return_value = mock_store

            from app.rag.ingestion import ingestion_pipeline

            # Should re-raise the transient error
            with pytest.raises(RuntimeError, match="Chroma crashed during insert"):
                await ingestion_pipeline.process_file(
                    pdf_path, "mod_fail.pdf", "application/pdf", "job-mod-fail", owner_id="user-1"
                )

            # Parents were added to store (data modified = True), so cache generation must be bumped
            mock_bump.assert_called_once_with("user-1")
