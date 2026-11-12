"""Unit tests for worker cache lifecycle and Celery failure policy (R-03, R-18)."""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.documents import Document

import app.core.cache as cache_module
from app.core.cache import init_cache, close_cache
from app.worker import celery_app, process_file_task


class FakeRedisAsync:
    """In-memory Redis fake implementing the async redis-py interface."""

    def __init__(self):
        self.store = {}
        self.closed = False

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: str):
        self.store[key] = value

    async def setex(self, key: str, ttl: int, value: str):
        self.store[key] = value

    async def delete(self, key: str):
        self.store.pop(key, None)

    async def ping(self):
        return True

    async def aclose(self):
        self.closed = True

    async def close(self):
        self.closed = True


class TestWorkerCacheAndFailurePolicy:
    def test_celery_config_settings_r18(self):
        """Celery config must contain reliability and limit settings (R-18)."""
        conf = celery_app.conf
        assert conf.task_acks_late is True
        assert conf.worker_prefetch_multiplier == 1
        assert conf.task_time_limit == 600
        assert conf.task_soft_time_limit == 540
        assert conf.result_expires == 86400

    @pytest.mark.asyncio
    async def test_init_cache_idempotency_does_not_leak_clients(self):
        """init_cache must be idempotent and close previous client on re-call (R-03)."""
        fake1 = FakeRedisAsync()
        fake2 = FakeRedisAsync()

        with patch.dict("os.environ", {"CACHE_BACKEND": "redis"}), \
             patch("redis.asyncio.from_url", side_effect=[fake1, fake2]):
            await init_cache()
            assert cache_module._redis_client is fake1
            assert fake1.closed is False

            # Calling init_cache again closes fake1 and sets fake2 without leaks
            await init_cache()
            assert fake1.closed is True
            assert cache_module._redis_client is fake2
            assert fake2.closed is False

            await close_cache()
            assert fake2.closed is True
            assert cache_module._redis_client is None

    def test_worker_task_initializes_cache_and_writes_final_job_status_in_redis(self, tmp_path):
        """R-03: With a fake Redis client, process_file_task sets job:{id} to completed via shared cache API."""
        fake_redis = FakeRedisAsync()
        pdf_path = str(tmp_path / "work_test.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 dummy")

        mock_docs = [Document(page_content="worker text", metadata={"filename": "work_test.pdf"})]

        with patch.dict("os.environ", {"CACHE_BACKEND": "redis"}), \
             patch("redis.asyncio.from_url", return_value=fake_redis), \
             patch("app.rag.ingestion.load_pdf", new_callable=AsyncMock, return_value=mock_docs), \
             patch("app.rag.ingestion.chunk_documents", return_value=(mock_docs, {"p1": "parent"})), \
             patch("app.rag.ingestion.embed_documents", new_callable=AsyncMock, return_value=[[0.1] * 384]), \
             patch("app.rag.ingestion.add_documents", new_callable=AsyncMock), \
             patch("app.rag.ingestion.get_bm25_store") as mock_bm25, \
             patch("app.rag.graph.graph_extractor.extract_and_store_graph", new_callable=AsyncMock):

            mock_store = MagicMock()
            mock_bm25.return_value = mock_store

            # Execute the Celery worker task directly
            process_file_task(
                file_path=pdf_path,
                filename="work_test.pdf",
                content_type="application/pdf",
                job_id="job-worker-1",
                owner_id="1",
                doc_id="1:doc-w1",
                replace=True,
            )

            # Verification via the fake Redis client: job:job-worker-1 should be completed
            stored_val = fake_redis.store.get("job:job-worker-1")
            assert stored_val is not None
            assert json.loads(stored_val) == "completed"

            # After task, client in module should be closed
            assert cache_module._redis_client is None
            assert fake_redis.closed is True

    def test_worker_task_transient_error_re_raises_for_celery_retry_and_keeps_file(self, tmp_path):
        """R-18: Transient errors re-raise so Celery retries, keeping upload file until last retry."""
        test_file = tmp_path / "transient.pdf"
        test_file.write_bytes(b"%PDF-1.4 transient")

        with patch("app.rag.ingestion.ingestion_pipeline.process_file", new_callable=AsyncMock) as mock_pf, \
             patch.object(process_file_task, "retry", side_effect=RuntimeError("Celery retry scheduled")) as mock_retry:
            mock_pf.side_effect = ConnectionError("Redis connection dropped")
            process_file_task.request.retries = 1
            process_file_task.max_retries = 3

            with pytest.raises(RuntimeError, match="Celery retry scheduled"):
                process_file_task.run(
                    file_path=str(test_file),
                    filename="transient.pdf",
                    content_type="application/pdf",
                    job_id="job-transient",
                    owner_id="1",
                )

            # Retry was called with exponential backoff (2 ** 1 = 2s)
            mock_retry.assert_called_once()
            _, kwargs = mock_retry.call_args
            assert kwargs.get("countdown") == 2

            # File was NOT deleted; kept for next retry attempt
            assert test_file.exists()

    def test_worker_task_exhausted_retries_marks_job_failed_and_deletes_file(self, tmp_path):
        """R-18: When retries are exhausted, job status becomes failed: and upload file is deleted."""
        test_file = tmp_path / "exhausted.pdf"
        test_file.write_bytes(b"%PDF-1.4 exhausted")

        with patch("app.rag.ingestion.ingestion_pipeline.process_file", new_callable=AsyncMock) as mock_pf, \
             patch("app.worker._set_job_failed_status", new_callable=AsyncMock) as mock_failed_status:
            mock_pf.side_effect = ConnectionError("DB connection dead")
            process_file_task.request.retries = 3
            process_file_task.max_retries = 3

            with pytest.raises(ConnectionError, match="DB connection dead"):
                process_file_task.run(
                    file_path=str(test_file),
                    filename="exhausted.pdf",
                    content_type="application/pdf",
                    job_id="job-exhausted",
                    owner_id="1",
                )

            # Job status marked failed
            mock_failed_status.assert_called_once()
            assert "DB connection dead" in mock_failed_status.call_args[0][1]

            # Upload file deleted
            assert not test_file.exists()
