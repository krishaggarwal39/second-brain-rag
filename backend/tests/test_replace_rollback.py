"""Unit tests for replace-on-upload ordering and rollback (R-19)."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.documents import Document

from app.rag.ingestion import ingestion_pipeline


class TestReplaceRollbackR19:
    @pytest.mark.asyncio
    async def test_replace_writes_new_version_before_deleting_old(self, tmp_path):
        """R-19: New version (vectors, BM25, graph, parents) is written first, then old version is deleted."""
        pdf_path = str(tmp_path / "replace_order.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 dummy")

        mock_docs = [Document(page_content="new doc text", metadata={"filename": "doc.pdf"})]
        old_meta = [{"doc_id": "user-1:old-id", "filename": "doc.pdf", "owner_id": "user-1"}]

        execution_order = []

        async def track_add_docs(*args, **kwargs):
            execution_order.append("add_documents")

        async def track_delete_doc(doc_id, *args, **kwargs):
            execution_order.append(f"delete_document:{doc_id}")

        with patch("app.rag.ingestion.load_pdf", new_callable=AsyncMock, return_value=mock_docs), \
             patch("app.rag.ingestion.chunk_documents", return_value=(mock_docs, {"p": "parent text"})), \
             patch("app.rag.ingestion.embed_documents", new_callable=AsyncMock, return_value=[[0.1] * 384]), \
             patch("app.rag.ingestion.add_documents", new_callable=AsyncMock, side_effect=track_add_docs), \
             patch("app.rag.ingestion.delete_document", new_callable=AsyncMock, side_effect=track_delete_doc), \
             patch("app.rag.ingestion.get_bm25_store") as mock_bm25, \
             patch("app.rag.graph.graph_extractor.extract_and_store_graph", new_callable=AsyncMock):

            mock_store = MagicMock()
            mock_store.get_document_metadata.return_value = old_meta
            mock_store.add_parents = MagicMock(side_effect=lambda *a, **k: execution_order.append("add_parents"))
            mock_store.add_documents = MagicMock(side_effect=lambda *a, **k: execution_order.append("bm25_add"))
            mock_store.delete_documents_by_doc_id = MagicMock(side_effect=lambda *a, **k: execution_order.append("bm25_delete_old"))
            mock_store.delete_parents_by_doc_id = MagicMock(side_effect=lambda *a, **k: execution_order.append("delete_parents_old"))
            mock_bm25.return_value = mock_store

            await ingestion_pipeline.process_file(
                pdf_path, "doc.pdf", "application/pdf", "job-rep-1", owner_id="user-1", replace=True
            )

            # Assert new version was written BEFORE old version was deleted
            assert "add_parents" in execution_order
            assert "add_documents" in execution_order
            assert "bm25_add" in execution_order

            idx_new_chroma = execution_order.index("add_documents")
            idx_del_old = execution_order.index("delete_document:user-1:old-id")
            assert idx_new_chroma < idx_del_old

    @pytest.mark.asyncio
    async def test_replace_failure_rolls_back_new_version_and_preserves_old(self, tmp_path):
        """R-19: On failure writing new version, rolls back new version and leaves old version untouched."""
        pdf_path = str(tmp_path / "rollback.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 dummy")

        mock_docs = [Document(page_content="failing doc text", metadata={"filename": "doc.pdf"})]
        old_meta = [{"doc_id": "user-1:old-id", "filename": "doc.pdf", "owner_id": "user-1"}]

        deleted_doc_ids = []

        async def track_delete_doc(doc_id, *args, **kwargs):
            deleted_doc_ids.append(doc_id)

        # Chroma add succeeds, but BM25 add_documents fails
        with patch("app.rag.ingestion.load_pdf", new_callable=AsyncMock, return_value=mock_docs), \
             patch("app.rag.ingestion.chunk_documents", return_value=(mock_docs, {"p": "parent text"})), \
             patch("app.rag.ingestion.embed_documents", new_callable=AsyncMock, return_value=[[0.1] * 384]), \
             patch("app.rag.ingestion.add_documents", new_callable=AsyncMock), \
             patch("app.rag.ingestion.delete_document", new_callable=AsyncMock, side_effect=track_delete_doc), \
             patch("app.rag.ingestion.get_bm25_store") as mock_bm25:

            mock_store = MagicMock()
            mock_store.get_document_metadata.return_value = old_meta
            mock_store.add_documents.side_effect = RuntimeError("BM25 SQLite disk full")
            mock_bm25.return_value = mock_store

            with pytest.raises(RuntimeError, match="BM25 SQLite disk full"):
                await ingestion_pipeline.process_file(
                    pdf_path, "doc.pdf", "application/pdf", "job-roll-1", owner_id="user-1", replace=True
                )

            # Old doc was NEVER deleted!
            assert "user-1:old-id" not in deleted_doc_ids

            # New doc_id WAS rolled back via delete_document
            assert len(deleted_doc_ids) == 1
            new_rolled_back_id = deleted_doc_ids[0]
            assert new_rolled_back_id != "user-1:old-id"
            assert new_rolled_back_id.startswith("user-1:")
