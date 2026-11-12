import asyncio
import os
import time
from typing import Awaitable, Callable, Dict, Optional

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])

CHECK_TIMEOUT_SECONDS = float(os.getenv("READY_CHECK_TIMEOUT", "2"))
LLM_CHECK_TTL_SECONDS = float(os.getenv("READY_LLM_CHECK_TTL", "60"))

# Informational LLM check, cached so unauthenticated probes cannot burn quota (R-31).
_llm_cache: Dict[str, object] = {"at": 0.0, "status": None}


@router.get("/health")
async def health_check():
    """Liveness: the process is up. Cheap, no dependencies."""
    return {"status": "ok"}


async def _check_chroma() -> None:
    from app.rag.vectorstore.chroma_store import get_client

    client = await asyncio.to_thread(get_client)
    await asyncio.to_thread(client.heartbeat)


async def _check_redis() -> None:
    from app.core import cache

    client = cache._redis_client
    if client is None:
        raise RuntimeError("redis client not initialised")
    await client.ping()


async def _check_database() -> None:
    from app.core.auth import get_user_store

    store = await asyncio.to_thread(get_user_store)
    await asyncio.to_thread(store.user_exists, 0)  # cheap indexed SELECT


async def _check_neo4j() -> None:
    from app.rag.graph.neo4j_manager import get_neo4j_manager

    manager = get_neo4j_manager()
    if not manager.driver:
        raise RuntimeError("driver not initialised")
    await asyncio.to_thread(manager.driver.verify_connectivity)


async def _run(name: str, fn: Callable[[], Awaitable[None]]) -> tuple[str, str]:
    try:
        await asyncio.wait_for(fn(), timeout=CHECK_TIMEOUT_SECONDS)
        return name, "ok"
    except asyncio.TimeoutError:
        logger.warning("Readiness check timed out", component=name)
        return name, "timeout"
    except Exception as e:
        logger.warning("Readiness check failed", component=name, error_type=type(e).__name__)
        return name, "unavailable"


async def _llm_status() -> Optional[str]:
    now = time.time()
    if _llm_cache["status"] is not None and now - float(_llm_cache["at"]) < LLM_CHECK_TTL_SECONDS:
        return _llm_cache["status"]  # type: ignore[return-value]
    key = (os.getenv("GOOGLE_API_KEY", "").split(",")[0]).strip()
    if not key:
        status = "not_configured"
    else:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    headers={"x-goog-api-key": key},
                    timeout=CHECK_TIMEOUT_SECONDS,
                )
            status = "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
        except Exception:
            status = "unreachable"
    _llm_cache.update(at=now, status=status)
    return status


@router.get("/health/ready")
async def readiness_check():
    """Readiness: required local dependencies answer within a short timeout.

    Returns 503 with per-component status when any required check fails. The
    LLM provider check is informational only (it never fails readiness).
    """
    checks: Dict[str, Callable[[], Awaitable[None]]] = {
        "chroma": _check_chroma,
        "database": _check_database,
    }
    if os.getenv("CACHE_BACKEND") == "redis":
        checks["redis"] = _check_redis
    if os.getenv("DISABLE_GRAPH", "0") != "1":
        checks["neo4j"] = _check_neo4j

    results = dict(await asyncio.gather(*(_run(n, f) for n, f in checks.items())))
    body = {
        "status": "ready" if all(v == "ok" for v in results.values()) else "unavailable",
        "components": results,
        "llm": await _llm_status() if os.getenv("READY_CHECK_LLM", "true").lower() == "true" else "skipped",
    }
    return JSONResponse(status_code=200 if body["status"] == "ready" else 503, content=body)
