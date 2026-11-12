import os
import time
import threading
from typing import List
from app.core.logging import get_logger

logger = get_logger(__name__)

_model = None
_model_lock = threading.Lock()
_last_failed_time = 0.0
COOLDOWN_SECONDS = 30.0

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))


def _get_model():
    """Lazy-load the SentenceTransformer model on first use (thread-safe with cooldown).

    R-28: Add lock around lazy model load and allow retry after a failed load
    (cooldown 30s) instead of failing permanently.
    """
    global _model, _last_failed_time
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        now = time.time()
        if (now - _last_failed_time) < COOLDOWN_SECONDS:
            logger.warning(
                "Embedding model load failed recently; in cooldown period",
                cooldown_remaining=round(COOLDOWN_SECONDS - (now - _last_failed_time), 1),
            )
            return None

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {EMBEDDING_MODEL}")
            _model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully.")
            _last_failed_time = 0.0
            return _model
        except Exception as e:
            logger.error(f"Failed to load local embedding model: {e}")
            _model = None
            _last_failed_time = time.time()
            return None


async def embed_documents(texts: List[str]) -> List[List[float]]:
    """Generate embeddings locally using SentenceTransformers.

    Runs the CPU-bound encode() in a thread pool so the async event loop
    is never blocked.
    """
    if not texts:
        return []

    try:
        import asyncio
        model = await asyncio.to_thread(_get_model)
        if model is None:
            logger.error("Embedding model unavailable — cannot embed documents.")
            return []

        embeddings = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]
            batch_embeddings = await asyncio.to_thread(model.encode, batch_texts, show_progress_bar=False)
            embeddings.extend(batch_embeddings.tolist())
        return embeddings
    except Exception as e:
        logger.error(f"Local embedding generation failed: {e}")
        return []
