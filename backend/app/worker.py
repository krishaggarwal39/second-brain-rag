import os
import asyncio
from typing import Optional
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
if REDIS_PASSWORD and "@" not in REDIS_URL:
    REDIS_URL = REDIS_URL.replace("redis://", f"redis://:{REDIS_PASSWORD}@")

celery_app = Celery(
    "second_brain_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Concurrency choice (R-18): Default 2 via env CELERY_CONCURRENCY.
# Each Celery worker child loads heavy embedding and reranking models into memory.
# Setting concurrency to 1-2 prevents out-of-memory errors on typical host instances
# while allowing concurrent I/O. For production compose, batch 5 configures CELERY_CONCURRENCY.
CELERY_CONCURRENCY = int(os.getenv("CELERY_CONCURRENCY", "2"))

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # High-reliability settings for large tasks (PDF parsing/LLMs)
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "600")),
    task_soft_time_limit=int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "540")),
    result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "86400")),
)


async def _execute_file_task(
    file_path: str,
    filename: str,
    content_type: str,
    job_id: str,
    owner_id: str,
    doc_id: Optional[str],
    replace: bool,
):
    """R-03: Initialize cache client at task start and close in finally."""
    from app.core.cache import init_cache, close_cache
    from app.rag.ingestion import ingestion_pipeline

    await init_cache()
    try:
        await ingestion_pipeline.process_file(
            file_path=file_path,
            filename=filename,
            content_type=content_type,
            job_id=job_id,
            owner_id=owner_id,
            doc_id=doc_id,
            replace=replace,
        )
    finally:
        await close_cache()


async def _execute_url_task(
    url: str,
    job_id: str,
    owner_id: str,
    doc_id: Optional[str],
    replace: bool,
):
    """R-03: Initialize cache client at task start and close in finally."""
    from app.core.cache import init_cache, close_cache
    from app.rag.ingestion import ingestion_pipeline

    await init_cache()
    try:
        await ingestion_pipeline.process_url(
            url=url,
            job_id=job_id,
            owner_id=owner_id,
            doc_id=doc_id,
            replace=replace,
        )
    finally:
        await close_cache()


async def _set_job_failed_status(job_id: str, message: str):
    """Record job failure status in cache when retries are exhausted (R-18)."""
    from app.core.cache import init_cache, close_cache, set_cache

    try:
        await init_cache()
        await set_cache(f"job:{job_id}", message)
    except Exception:
        pass
    finally:
        await close_cache()


@celery_app.task(bind=True, max_retries=3)
def process_file_task(
    self,
    file_path: str,
    filename: str,
    content_type: str,
    job_id: str,
    owner_id: str = "",
    doc_id: Optional[str] = None,
    replace: bool = True,
):
    from app.core.upload import cleanup_old_uploads
    cleanup_old_uploads()

    try:
        asyncio.run(
            _execute_file_task(
                file_path=file_path,
                filename=filename,
                content_type=content_type,
                job_id=job_id,
                owner_id=owner_id,
                doc_id=doc_id,
                replace=replace,
            )
        )
    except Exception as exc:
        # R-18: Check if retries are exhausted
        if self.request.retries >= self.max_retries:
            asyncio.run(_set_job_failed_status(job_id, f"failed: {str(exc)}"))
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            raise
        # R-18: Keep file until last attempt, retry with exponential backoff
        countdown = 2 ** self.request.retries
        raise self.retry(exc=exc, countdown=countdown)


@celery_app.task(bind=True, max_retries=3)
def process_url_task(
    self,
    url: str,
    job_id: str,
    owner_id: str = "",
    doc_id: Optional[str] = None,
    replace: bool = True,
):
    from app.core.upload import cleanup_old_uploads
    cleanup_old_uploads()

    try:
        asyncio.run(
            _execute_url_task(
                url=url,
                job_id=job_id,
                owner_id=owner_id,
                doc_id=doc_id,
                replace=replace,
            )
        )
    except Exception as exc:
        # R-18: Check if retries are exhausted
        if self.request.retries >= self.max_retries:
            asyncio.run(_set_job_failed_status(job_id, f"failed: {str(exc)}"))
            raise
        countdown = 2 ** self.request.retries
        raise self.retry(exc=exc, countdown=countdown)
