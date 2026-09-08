"""Tests for the ingestion pipeline."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.documents import Document


class TestIngestionPipeline:
    @pytest.mark.asyncio
    async def test_process_file_pdf(self, tmp_path):
        """Test that PDF ingestion flows through the full pipeline."""
        pdf_path = str(tmp_path / "test.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"fake pdf content")

        mock_docs = [
            Document(
                page_content="Test content from PDF",
                metadata={"filename": "test.pdf", "page_number": 1, "source_type": "pdf"},
            )
        ]

        with patch(
            "app.rag.ingestion.load_pdf", new_callable=AsyncMock, return_value=mock_docs
        ), patch(
            "app.rag.ingestion.chunk_documents",
            return_value=(mock_docs, {"parent-1": "parent text"}),
        ), patch(
            "app.rag.ingestion.embed_documents",
            new_callable=AsyncMock,
            return_value=[[0.1] * 384],
        ), patch(
            "app.rag.ingestion.add_documents", new_callable=AsyncMock
        ), patch(
            "app.rag.ingestion.set_cache", new_callable=AsyncMock
        ) as mock_cache, patch(
            "app.rag.ingestion.get_bm25_store"
        ) as mock_bm25, patch(
            "app.rag.graph.graph_extractor.extract_and_store_graph", new_callable=AsyncMock
        ):
            mock_store = MagicMock()
            mock_store.add_documents = MagicMock()
            mock_store.update_document_metadata = MagicMock()
            mock_bm25.return_value = mock_store

            from app.rag.ingestion import ingestion_pipeline

            await ingestion_pipeline.process_file(
                pdf_path, "test.pdf", "application/pdf", "job-123", owner_id="user-1"
            )

            # Verify job status was set to completed
            mock_cache.assert_any_call("job:job-123", "completed")
            mock_store.update_document_metadata.assert_called_once()
            _, kwargs = mock_store.update_document_metadata.call_args
            assert kwargs.get("owner_id") == "user-1"
            assert kwargs.get("filename") == "test.pdf"
            assert kwargs.get("doc_id") is not None
            assert kwargs.get("doc_id").startswith("user-1:")

            mock_store.add_documents.assert_called_once()
            bm25_docs = mock_store.add_documents.call_args[0][0]
            assert len(bm25_docs) == 1
            assert bm25_docs[0]["owner_id"] == "user-1"
            assert bm25_docs[0]["doc_id"] == kwargs["doc_id"]

    @pytest.mark.asyncio
    async def test_process_file_unsupported_format(self, tmp_path):
        """Unsupported formats should mark the job as failed."""
        txt_path = str(tmp_path / "test.txt")
        with open(txt_path, "w") as f:
            f.write("plain text")

        with patch(
            "app.rag.ingestion.set_cache", new_callable=AsyncMock
        ) as mock_cache:
            from app.rag.ingestion import ingestion_pipeline

            await ingestion_pipeline.process_file(
                txt_path, "test.txt", "text/plain", "job-456", owner_id="user-1"
            )

            # Should be marked as failed
            calls = [str(c) for c in mock_cache.call_args_list]
            assert any("failed" in c for c in calls)

    @pytest.mark.asyncio
    async def test_process_url(self):
        """Test URL ingestion calls the web loader."""
        mock_docs = [
            Document(
                page_content="Web page content",
                metadata={"url": "https://example.com", "source_type": "web"},
            )
        ]

        with patch(
            "app.rag.ingestion.load_web", new_callable=AsyncMock, return_value=mock_docs
        ), patch(
            "app.rag.ingestion.chunk_documents",
            return_value=(mock_docs, {"p1": "parent"}),
        ), patch(
            "app.rag.ingestion.embed_documents",
            new_callable=AsyncMock,
            return_value=[[0.1] * 384],
        ), patch(
            "app.rag.ingestion.add_documents", new_callable=AsyncMock
        ), patch(
            "app.rag.ingestion.set_cache", new_callable=AsyncMock
        ) as mock_cache, patch(
            "app.rag.ingestion.get_bm25_store"
        ) as mock_bm25, patch(
            "app.rag.graph.graph_extractor.extract_and_store_graph", new_callable=AsyncMock
        ):
            mock_store = MagicMock()
            mock_store.add_documents = MagicMock()
            mock_store.update_document_metadata = MagicMock()
            mock_bm25.return_value = mock_store

            from app.rag.ingestion import ingestion_pipeline

            await ingestion_pipeline.process_url("https://example.com", "job-789", owner_id="user-1")

            mock_cache.assert_any_call("job:job-789", "completed")
            mock_store.update_document_metadata.assert_called_once()
            _, kwargs = mock_store.update_document_metadata.call_args
            assert kwargs.get("owner_id") == "user-1"
            assert kwargs.get("filename") == "https://example.com"
            assert kwargs.get("doc_id") is not None
            assert kwargs.get("doc_id").startswith("user-1:")

            mock_store.add_documents.assert_called_once()
            bm25_docs = mock_store.add_documents.call_args[0][0]
            assert len(bm25_docs) == 1
            assert bm25_docs[0]["owner_id"] == "user-1"
            assert bm25_docs[0]["doc_id"] == kwargs["doc_id"]

    @pytest.mark.asyncio
    async def test_cleanup_temp_file_on_success(self, tmp_path):
        """Temp files should be removed after processing."""
        import os

        pdf_path = str(tmp_path / "cleanup_test.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"content")

        mock_docs = [
            Document(page_content="text", metadata={"filename": "f.pdf", "source_type": "pdf"})
        ]

        with patch(
            "app.rag.ingestion.load_pdf", new_callable=AsyncMock, return_value=mock_docs
        ), patch(
            "app.rag.ingestion.chunk_documents",
            return_value=(mock_docs, {"p": "text"}),
        ), patch(
            "app.rag.ingestion.embed_documents",
            new_callable=AsyncMock,
            return_value=[[0.1] * 384],
        ), patch(
            "app.rag.ingestion.add_documents", new_callable=AsyncMock
        ), patch(
            "app.rag.ingestion.set_cache", new_callable=AsyncMock
        ), patch(
            "app.rag.ingestion.get_bm25_store"
        ) as mock_bm25, patch(
            "app.rag.graph.graph_extractor.extract_and_store_graph", new_callable=AsyncMock
        ):
            mock_store = MagicMock()
            mock_store.add_documents = MagicMock()
            mock_store.update_document_metadata = MagicMock()
            mock_bm25.return_value = mock_store

            from app.rag.ingestion import ingestion_pipeline

            await ingestion_pipeline.process_file(
                pdf_path, "cleanup_test.pdf", "application/pdf", "job-cleanup", owner_id="user-1"
            )

            # File should be deleted
            assert not os.path.exists(pdf_path)

    @pytest.mark.asyncio
    async def test_process_file_with_explicit_doc_id(self, tmp_path):
        """Test that explicit doc_id is preserved through ingestion."""
        pdf_path = str(tmp_path / "explicit.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"content")

        mock_docs = [
            Document(page_content="text", metadata={"filename": "explicit.pdf", "source_type": "pdf"})
        ]

        with patch(
            "app.rag.ingestion.load_pdf", new_callable=AsyncMock, return_value=mock_docs
        ), patch(
            "app.rag.ingestion.chunk_documents",
            return_value=(mock_docs, {"p": "text"}),
        ), patch(
            "app.rag.ingestion.embed_documents",
            new_callable=AsyncMock,
            return_value=[[0.1] * 384],
        ), patch(
            "app.rag.ingestion.add_documents", new_callable=AsyncMock
        ), patch(
            "app.rag.ingestion.set_cache", new_callable=AsyncMock
        ), patch(
            "app.rag.ingestion.get_bm25_store"
        ) as mock_bm25, patch(
            "app.rag.graph.graph_extractor.extract_and_store_graph", new_callable=AsyncMock
        ):
            mock_store = MagicMock()
            mock_store.add_documents = MagicMock()
            mock_store.update_document_metadata = MagicMock()
            mock_bm25.return_value = mock_store

            from app.rag.ingestion import ingestion_pipeline

            custom_doc_id = "user-99:custom-doc-uuid"
            await ingestion_pipeline.process_file(
                pdf_path, "explicit.pdf", "application/pdf", "job-custom",
                owner_id="user-99", doc_id=custom_doc_id
            )

            _, kwargs = mock_store.update_document_metadata.call_args
            assert kwargs.get("doc_id") == custom_doc_id
            assert kwargs.get("filename") == "explicit.pdf"
            assert kwargs.get("owner_id") == "user-99"

            bm25_docs = mock_store.add_documents.call_args[0][0]
            assert bm25_docs[0]["doc_id"] == custom_doc_id
            assert bm25_docs[0]["owner_id"] == "user-99"

    @pytest.mark.asyncio
    async def test_cache_generation_bumped_on_completed_ingestion(self, tmp_path):
        """Cache generation should be bumped after process_file and process_url complete."""
        pdf_path = str(tmp_path / "gen_test.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"content")

        mock_docs = [Document(page_content="hello", metadata={"filename": "gen_test.pdf"})]

        with patch("app.rag.ingestion.load_pdf", new_callable=AsyncMock, return_value=mock_docs), \
             patch("app.rag.ingestion.load_web", new_callable=AsyncMock, return_value=mock_docs), \
             patch("app.rag.ingestion.chunk_documents", return_value=(mock_docs, {})), \
             patch("app.rag.ingestion.embed_documents", new_callable=AsyncMock, return_value=[[0.1] * 384]), \
             patch("app.rag.ingestion.add_documents", new_callable=AsyncMock), \
             patch("app.rag.ingestion.set_cache", new_callable=AsyncMock), \
             patch("app.rag.ingestion.get_bm25_store"), \
             patch("app.rag.graph.graph_extractor.extract_and_store_graph", new_callable=AsyncMock), \
             patch("app.rag.ingestion.bump_cache_generation", new_callable=AsyncMock) as mock_bump:

            from app.rag.ingestion import ingestion_pipeline

            # 1. process_file bumps generation for owner-1
            await ingestion_pipeline.process_file(
                pdf_path, "gen_test.pdf", "application/pdf", "job-f1", owner_id="owner-1"
            )
            mock_bump.assert_called_once_with("owner-1")

            mock_bump.reset_mock()

            # 2. process_url bumps generation for owner-2
            await ingestion_pipeline.process_url(
                "https://example.com/test", "job-u1", owner_id="owner-2"
            )
            mock_bump.assert_called_once_with("owner-2")

    @pytest.mark.asyncio
    async def test_process_file_embedding_failure_empty(self, tmp_path):
        """When embed_documents returns [], ingestion must fail and not partially index BM25."""
        import os
        pdf_path = str(tmp_path / "embed_fail.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"content")

        mock_docs = [Document(page_content="some content", metadata={"filename": "embed_fail.pdf"})]

        with patch("app.rag.ingestion.load_pdf", new_callable=AsyncMock, return_value=mock_docs), \
             patch("app.rag.ingestion.chunk_documents", return_value=(mock_docs, {"p1": "parent text"})), \
             patch("app.rag.ingestion.embed_documents", new_callable=AsyncMock, return_value=[]), \
             patch("app.rag.ingestion.add_documents", new_callable=AsyncMock) as mock_chroma_add, \
             patch("app.rag.ingestion.set_cache", new_callable=AsyncMock) as mock_cache, \
             patch("app.rag.ingestion.get_bm25_store") as mock_bm25, \
             patch("app.rag.graph.graph_extractor.extract_and_store_graph", new_callable=AsyncMock) as mock_graph:

            mock_store = MagicMock()
            mock_bm25.return_value = mock_store

            from app.rag.ingestion import ingestion_pipeline

            await ingestion_pipeline.process_file(
                pdf_path, "embed_fail.pdf", "application/pdf", "job-embed-fail", owner_id="user-1"
            )

            # Failure should be reported in job cache
            mock_cache.assert_any_call("job:job-embed-fail", "failed: embedding generation failed")
            # Should NOT mark as completed
            for call in mock_cache.call_args_list:
                assert call.args != ("job:job-embed-fail", "completed")

            # Must not add to Chroma or BM25
            mock_chroma_add.assert_not_called()
            mock_store.add_documents.assert_not_called()
            mock_store.update_document_metadata.assert_not_called()
            mock_graph.assert_not_called()

            # Temp file is still cleaned up
            assert not os.path.exists(pdf_path)

    @pytest.mark.asyncio
    async def test_process_file_embedding_length_mismatch_fails(self, tmp_path):
        """When embed_documents returns fewer embeddings than texts, fail cleanly."""
        pdf_path = str(tmp_path / "embed_mismatch.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"content")

        mock_chunks = [
            Document(page_content="chunk 1", metadata={"filename": "embed_mismatch.pdf"}),
            Document(page_content="chunk 2", metadata={"filename": "embed_mismatch.pdf"}),
        ]

        with patch("app.rag.ingestion.load_pdf", new_callable=AsyncMock, return_value=mock_chunks), \
             patch("app.rag.ingestion.chunk_documents", return_value=(mock_chunks, {"p1": "parent"})), \
             patch("app.rag.ingestion.embed_documents", new_callable=AsyncMock, return_value=[[0.1] * 384]), \
             patch("app.rag.ingestion.add_documents", new_callable=AsyncMock) as mock_chroma_add, \
             patch("app.rag.ingestion.set_cache", new_callable=AsyncMock) as mock_cache, \
             patch("app.rag.ingestion.get_bm25_store") as mock_bm25:

            mock_store = MagicMock()
            mock_bm25.return_value = mock_store

            from app.rag.ingestion import ingestion_pipeline

            await ingestion_pipeline.process_file(
                pdf_path, "embed_mismatch.pdf", "application/pdf", "job-mismatch", owner_id="user-1"
            )

            mock_cache.assert_any_call("job:job-mismatch", "failed: embedding generation failed")
            mock_chroma_add.assert_not_called()
            mock_store.add_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_graph_extraction_uses_parent_chunks_and_configurable_cap(self, tmp_path, monkeypatch):
        """Graph extraction must extract from parent chunks, not child chunks, and respect GRAPH_MAX_SEGMENTS."""
        pdf_path = str(tmp_path / "graph_test.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"content")

        mock_chunks = [
            Document(page_content=f"child chunk {i}", metadata={"filename": "graph_test.pdf"})
            for i in range(5)
        ]
        parent_store = {
            f"parent_{i}": f"Full parent text number {i}"
            for i in range(5)
        }

        monkeypatch.setenv("GRAPH_MAX_SEGMENTS", "3")

        with patch("app.rag.ingestion.load_pdf", new_callable=AsyncMock, return_value=mock_chunks), \
             patch("app.rag.ingestion.chunk_documents", return_value=(mock_chunks, parent_store)), \
             patch("app.rag.ingestion.embed_documents", new_callable=AsyncMock, return_value=[[0.1] * 384] * 5), \
             patch("app.rag.ingestion.add_documents", new_callable=AsyncMock), \
             patch("app.rag.ingestion.set_cache", new_callable=AsyncMock), \
             patch("app.rag.ingestion.get_bm25_store") as mock_bm25, \
             patch("app.rag.graph.graph_extractor.extract_and_store_graph", new_callable=AsyncMock) as mock_graph, \
             patch("asyncio.sleep", new_callable=AsyncMock):

            mock_store = MagicMock()
            mock_bm25.return_value = mock_store

            from app.rag.ingestion import ingestion_pipeline

            await ingestion_pipeline.process_file(
                pdf_path, "graph_test.pdf", "application/pdf", "job-graph", owner_id="user-1"
            )

            # Cap was 3, so extract_and_store_graph should be called exactly 3 times
            assert mock_graph.call_count == 3
            # And calls should use the parent texts, NOT child chunks!
            extracted_texts = [call.args[0] for call in mock_graph.call_args_list]
            assert extracted_texts == [
                "Full parent text number 0",
                "Full parent text number 1",
                "Full parent text number 2",
            ]

    @pytest.mark.asyncio
    async def test_graph_extraction_failure_is_non_fatal(self, tmp_path):
        """If graph extraction fails, ingestion must still succeed and complete."""
        pdf_path = str(tmp_path / "graph_fail.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"content")

        mock_chunks = [Document(page_content="child", metadata={"filename": "graph_fail.pdf"})]

        with patch("app.rag.ingestion.load_pdf", new_callable=AsyncMock, return_value=mock_chunks), \
             patch("app.rag.ingestion.chunk_documents", return_value=(mock_chunks, {"p": "parent"})), \
             patch("app.rag.ingestion.embed_documents", new_callable=AsyncMock, return_value=[[0.1] * 384]), \
             patch("app.rag.ingestion.add_documents", new_callable=AsyncMock), \
             patch("app.rag.ingestion.set_cache", new_callable=AsyncMock) as mock_cache, \
             patch("app.rag.ingestion.get_bm25_store") as mock_bm25, \
             patch("app.rag.graph.graph_extractor.extract_and_store_graph", side_effect=RuntimeError("Neo4j down")), \
             patch("asyncio.sleep", new_callable=AsyncMock):

            mock_store = MagicMock()
            mock_bm25.return_value = mock_store

            from app.rag.ingestion import ingestion_pipeline

            await ingestion_pipeline.process_file(
                pdf_path, "graph_fail.pdf", "application/pdf", "job-graph-fail", owner_id="user-1"
            )

            # Ingestion should still complete despite graph failure
            mock_cache.assert_any_call("job:job-graph-fail", "completed")


