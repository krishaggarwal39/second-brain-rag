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
    backend=REDIS_URL
)

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
)


@celery_app.task(bind=True, max_retries=3)
def process_file_task(
    self,
    file_path: str,
    filename: str,
    content_type: str,
    job_id: str,
    owner_id: str = "",
    doc_id: Optional[str] = None,
):
    from app.rag.ingestion import ingestion_pipeline
    try:
        asyncio.run(
            ingestion_pipeline.process_file(
                file_path=file_path,
                filename=filename,
                content_type=content_type,
                job_id=job_id,
                owner_id=owner_id,
                doc_id=doc_id,
            )
        )
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)

@celery_app.task(bind=True, max_retries=3)
def process_url_task(
    self,
    url: str,
    job_id: str,
    owner_id: str = "",
    doc_id: Optional[str] = None,
):
    from app.rag.ingestion import ingestion_pipeline
    try:
        asyncio.run(
            ingestion_pipeline.process_url(
                url=url,
                job_id=job_id,
                owner_id=owner_id,
                doc_id=doc_id,
            )
        )
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
