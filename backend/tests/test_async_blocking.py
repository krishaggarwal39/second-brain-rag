"""Unit tests verifying non-blocking async execution for auth and BM25 writes (R-25)."""

import asyncio
import pytest
from unittest.mock import patch, MagicMock
from fastapi.security import HTTPAuthorizationCredentials

from app.core.auth import get_current_user
from app.rag.ingestion import IngestionPipeline
from langchain_core.documents import Document


class TestAsyncBlockingR25:
    @pytest.mark.asyncio
    async def test_get_current_user_uses_asyncio_to_thread_for_user_exists(self):
        """R-25: get_current_user runs synchronous user_exists via asyncio.to_thread."""
        creds = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="dummy.token"
        )

        with patch("app.core.auth._decode_token", return_value={"sub": "42"}), \
             patch("app.core.auth.get_user_store") as mock_get_store, \
             patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy_to_thread:

            mock_store = MagicMock()
            mock_store.user_exists.return_value = True
            mock_get_store.return_value = mock_store

            user_id = await get_current_user(creds)
            assert user_id == 42

            # Verify asyncio.to_thread was called with store.user_exists
            spy_to_thread.assert_any_call(mock_store.user_exists, 42)

    @pytest.mark.asyncio
    async def test_ingestion_bm25_writes_wrapped_in_to_thread(self):
        """R-25: Ingestion wraps synchronous BM25 writes in asyncio.to_thread."""
        mock_docs = [Document(page_content="text", metadata={"filename": "doc.pdf"})]

        with patch("app.rag.ingestion.chunk_documents", return_value=(mock_docs, {"p": "parent text"})), \
             patch("app.rag.ingestion.embed_documents", return_value=[[0.1] * 384]), \
             patch("app.rag.ingestion.add_documents"), \
             patch("app.rag.ingestion.get_bm25_store") as mock_bm25, \
             patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy_to_thread, \
             patch("app.rag.graph.graph_extractor.extract_and_store_graph"):

            mock_store = MagicMock()
            mock_store.get_document_metadata.return_value = []
            mock_bm25.return_value = mock_store

            await IngestionPipeline._process_docs(
                docs=mock_docs,
                job_id="job-async-1",
                doc_id="1:doc-async",
                owner_id="1",
                filename="doc.pdf",
                replace=False,
            )

            # Verify BM25 store add_documents and update_document_metadata were called via asyncio.to_thread
            called_funcs = [call.args[0] for call in spy_to_thread.call_args_list]
            assert mock_store.add_documents in called_funcs
            assert mock_store.update_document_metadata in called_funcs
            assert mock_store.add_parents in called_funcs
