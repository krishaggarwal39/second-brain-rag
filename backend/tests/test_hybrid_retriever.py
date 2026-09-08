"""Tests for the hybrid retrieval pipeline (RRF fusion and reranking)."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.rag.retrievers.hybrid_retriever import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_basic_fusion(self):
        dense = [
            {"id": "doc1", "text": "Machine learning", "score": 0.9},
            {"id": "doc2", "text": "Deep learning", "score": 0.8},
        ]
        sparse = [
            {"id": "doc2", "text": "Deep learning", "score": 3.5},
            {"id": "doc3", "text": "Neural networks", "score": 2.0},
        ]

        fused = reciprocal_rank_fusion(dense, sparse)

        # doc2 appears in both, should rank highest
        assert fused[0]["id"] == "doc2"
        assert len(fused) == 3  # doc1, doc2, doc3

    def test_empty_inputs(self):
        fused = reciprocal_rank_fusion([], [])
        assert fused == []

    def test_single_source(self):
        dense = [
            {"id": "a", "text": "only dense", "score": 0.9},
        ]
        fused = reciprocal_rank_fusion(dense, [])
        assert len(fused) == 1
        assert fused[0]["id"] == "a"

    def test_no_duplicate_docs_in_output(self):
        dense = [
            {"id": "shared", "text": "same doc", "score": 0.9},
        ]
        sparse = [
            {"id": "shared", "text": "same doc", "score": 5.0},
        ]
        fused = reciprocal_rank_fusion(dense, sparse)
        assert len(fused) == 1

    def test_assigns_ids_from_metadata_hash(self):
        """Documents without explicit id should get id from metadata hash."""
        dense = [
            {"text": "no id doc", "metadata": {"hash": "hash123"}, "score": 0.5},
        ]
        fused = reciprocal_rank_fusion(dense, [])
        assert fused[0]["id"] == "hash123"

    def test_custom_k_parameter(self):
        dense = [{"id": "a", "text": "doc", "score": 0.9}]
        sparse = [{"id": "b", "text": "doc2", "score": 1.0}]

        # With a very large k, scores should be very small but still work
        fused = reciprocal_rank_fusion(dense, sparse, k=1000)
        assert len(fused) == 2

    def test_preserves_document_content(self):
        """Fusion should preserve all original fields in documents."""
        dense = [
            {
                "id": "doc1",
                "text": "Full text content",
                "metadata": {"filename": "test.pdf", "page": 1},
                "score": 0.85,
            }
        ]
        fused = reciprocal_rank_fusion(dense, [])
        assert fused[0]["text"] == "Full text content"
        assert fused[0]["metadata"]["filename"] == "test.pdf"


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_hybrid_search_combines_sources(
        self, mock_chroma_results, mock_bm25_results
    ):
        with patch(
            "app.rag.retrievers.hybrid_retriever.embed_documents",
            new_callable=AsyncMock,
            return_value=[[0.1] * 384],
        ), patch(
            "app.rag.retrievers.hybrid_retriever.chroma_query",
            new_callable=AsyncMock,
            return_value=mock_chroma_results,
        ), patch(
            "app.rag.retrievers.hybrid_retriever.get_bm25_store"
        ) as mock_bm25_store, patch(
            "app.rag.retrievers.hybrid_retriever.reranker", None
        ):
            mock_store = MagicMock()
            mock_store.search.return_value = mock_bm25_results
            mock_bm25_store.return_value = mock_store

            from app.rag.retrievers.hybrid_retriever import hybrid_search

            results = await hybrid_search("machine learning", top_k=5)

            assert len(results) > 0
            assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_handles_empty_embeddings(self):
        """If embedding fails, should still return BM25 results."""
        with patch(
            "app.rag.retrievers.hybrid_retriever.embed_documents",
            new_callable=AsyncMock,
            return_value=[],
        ), patch(
            "app.rag.retrievers.hybrid_retriever.get_bm25_store"
        ) as mock_bm25_store, patch(
            "app.rag.retrievers.hybrid_retriever.reranker", None
        ):
            mock_store = MagicMock()
            mock_store.search.return_value = [
                {"id": "bm25-1", "text": "fallback result", "score": 2.0}
            ]
            mock_bm25_store.return_value = mock_store

            from app.rag.retrievers.hybrid_retriever import hybrid_search

            results = await hybrid_search("test query", top_k=3)
            assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_hybrid_search_with_owner_id(self):
        """hybrid_search with owner_id should filter Chroma by owner_id and exclude sparse docs of other owners."""
        owner = "user-42"
        dense_results = [
            {
                "id": "doc-dense-1",
                "text": "Dense chunk for user 42",
                "metadata": {"filename": "owned.pdf", "page_number": 3, "owner_id": owner},
                "score": 0.9,
            }
        ]
        bm25_results = [
            {"id": "doc-sparse-owned", "text": "Sparse chunk owned by 42", "score": 3.0},
            {"id": "doc-sparse-other", "text": "Sparse chunk of user 99", "score": 4.0},
        ]
        chunk_metas = {
            "doc-sparse-owned": {"filename": "sparse.pdf", "page_number": 7, "owner_id": owner},
            "doc-sparse-other": {"filename": "secret.pdf", "page_number": 1, "owner_id": "user-99"},
        }

        mock_chroma_query = AsyncMock(return_value=dense_results)
        mock_get_chunk_metas = AsyncMock(return_value=chunk_metas)

        with patch(
            "app.rag.retrievers.hybrid_retriever.embed_documents",
            new_callable=AsyncMock,
            return_value=[[0.1] * 384],
        ), patch(
            "app.rag.retrievers.hybrid_retriever.chroma_query",
            mock_chroma_query,
        ), patch(
            "app.rag.retrievers.hybrid_retriever.get_chunk_metadatas",
            mock_get_chunk_metas,
        ), patch(
            "app.rag.retrievers.hybrid_retriever.get_bm25_store"
        ) as mock_bm25_store, patch(
            "app.rag.retrievers.hybrid_retriever.reranker", None
        ):
            mock_store = MagicMock()
            mock_store.search.return_value = bm25_results
            mock_bm25_store.return_value = mock_store

            from app.rag.retrievers.hybrid_retriever import hybrid_search

            results = await hybrid_search("query text", top_k=5, owner_id=owner)

            # Chroma query must have received owner_id filter
            mock_chroma_query.assert_called_once()
            _, kwargs = mock_chroma_query.call_args
            assert kwargs.get("filters") == {"owner_id": owner}

            # BM25 sparse store search must have received owner_id
            mock_store.search.assert_called_once_with("query text", 15, owner_id=owner)

            # Only user-42 documents should be in results
            result_ids = [r["id"] for r in results]
            assert "doc-dense-1" in result_ids
            assert "doc-sparse-owned" in result_ids
            assert "doc-sparse-other" not in result_ids

            # Verify page_number is preserved
            for r in results:
                assert r["metadata"]["owner_id"] == owner
                assert r["page_number"] in (3, 7)

    @pytest.mark.asyncio
    async def test_hybrid_search_preserves_page_number_with_reranker(self):
        """Page number and metadata should survive CrossEncoder reranking."""
        dense_results = [
            {
                "id": "doc-1",
                "text": "Neural networks doc",
                "metadata": {"filename": "ai.pdf", "page_number": 5, "owner_id": "1"},
                "score": 0.88,
            }
        ]
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.99]

        with patch(
            "app.rag.retrievers.hybrid_retriever.embed_documents",
            new_callable=AsyncMock,
            return_value=[[0.1] * 384],
        ), patch(
            "app.rag.retrievers.hybrid_retriever.chroma_query",
            new_callable=AsyncMock,
            return_value=dense_results,
        ), patch(
            "app.rag.retrievers.hybrid_retriever.get_bm25_store"
        ) as mock_bm25_store, patch(
            "app.rag.retrievers.hybrid_retriever.reranker", mock_reranker
        ):
            mock_store = MagicMock()
            mock_store.search.return_value = []
            mock_bm25_store.return_value = mock_store

            from app.rag.retrievers.hybrid_retriever import hybrid_search

            results = await hybrid_search("neural", top_k=1, owner_id="1")
            assert len(results) == 1
            assert results[0]["page_number"] == 5
            assert results[0]["metadata"]["page_number"] == 5
            assert results[0]["filename"] == "ai.pdf"

    @pytest.mark.asyncio
    async def test_bm25_only_match_hydrates_page_number_and_filename_non_reranked(self):
        """BM25-only matches should have page_number and filename copied to top level without reranker."""
        sparse_results = [
            {"id": "sparse-chunk-1", "text": "BM25 only chunk text", "score": 3.0}
        ]
        chunk_metas = {
            "sparse-chunk-1": {"filename": "manual.pdf", "page_number": 9, "owner_id": "user-1"}
        }

        with patch("app.rag.retrievers.hybrid_retriever.embed_documents", new_callable=AsyncMock, return_value=[]), \
             patch("app.rag.retrievers.hybrid_retriever.chroma_query", new_callable=AsyncMock, return_value=[]), \
             patch("app.rag.retrievers.hybrid_retriever.get_chunk_metadatas", new_callable=AsyncMock, return_value=chunk_metas), \
             patch("app.rag.retrievers.hybrid_retriever.get_bm25_store") as mock_bm25_store, \
             patch("app.rag.retrievers.hybrid_retriever.reranker", None):

            mock_store = MagicMock()
            mock_store.search.return_value = sparse_results
            mock_bm25_store.return_value = mock_store

            from app.rag.retrievers.hybrid_retriever import hybrid_search

            results = await hybrid_search("manual", top_k=1, owner_id="user-1")
            assert len(results) == 1
            assert results[0]["filename"] == "manual.pdf"
            assert results[0]["page_number"] == 9

    @pytest.mark.asyncio
    async def test_bm25_only_match_hydrates_page_number_and_filename_reranked(self):
        """BM25-only matches should have page_number and filename copied to top level with CrossEncoder."""
        sparse_results = [
            {"id": "sparse-chunk-1", "text": "BM25 only chunk text", "score": 3.0}
        ]
        chunk_metas = {
            "sparse-chunk-1": {"filename": "guide.pdf", "page_number": 42, "owner_id": "user-2"}
        }
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.85]

        with patch("app.rag.retrievers.hybrid_retriever.embed_documents", new_callable=AsyncMock, return_value=[]), \
             patch("app.rag.retrievers.hybrid_retriever.chroma_query", new_callable=AsyncMock, return_value=[]), \
             patch("app.rag.retrievers.hybrid_retriever.get_chunk_metadatas", new_callable=AsyncMock, return_value=chunk_metas), \
             patch("app.rag.retrievers.hybrid_retriever.get_bm25_store") as mock_bm25_store, \
             patch("app.rag.retrievers.hybrid_retriever.reranker", mock_reranker):

            mock_store = MagicMock()
            mock_store.search.return_value = sparse_results
            mock_bm25_store.return_value = mock_store

            from app.rag.retrievers.hybrid_retriever import hybrid_search

            results = await hybrid_search("guide", top_k=1, owner_id="user-2")
            assert len(results) == 1
            assert results[0]["filename"] == "guide.pdf"
            assert results[0]["page_number"] == 42

