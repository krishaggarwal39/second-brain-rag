"""R-31: liveness is cheap; readiness checks local dependencies; LLM check is cached."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.routes import health


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    health._llm_cache.update(at=0.0, status=None)
    monkeypatch.setenv("DISABLE_GRAPH", "1")
    monkeypatch.setenv("CACHE_BACKEND", "memory")


async def _ok():
    return None


async def _boom():
    raise ConnectionError("down")


async def _slow():
    await asyncio.sleep(10)


def _resp(code):
    r = MagicMock()
    r.status_code = code
    return r


@pytest.mark.asyncio
async def test_liveness():
    assert await health.health_check() == {"status": "ok"}


@pytest.mark.asyncio
async def test_ready_when_all_ok():
    with patch.object(health, "_check_chroma", _ok), patch.object(health, "_check_database", _ok), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=_resp(200)):
        r = await health.readiness_check()
    assert r.status_code == 200
    assert b'"chroma":"ok"' in r.body and b'"database":"ok"' in r.body


@pytest.mark.asyncio
async def test_503_lists_failed_component():
    with patch.object(health, "_check_chroma", _boom), patch.object(health, "_check_database", _ok), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=_resp(200)):
        r = await health.readiness_check()
    assert r.status_code == 503
    assert b'"chroma":"unavailable"' in r.body


@pytest.mark.asyncio
async def test_timeout_reported(monkeypatch):
    monkeypatch.setattr(health, "CHECK_TIMEOUT_SECONDS", 0.05)
    with patch.object(health, "_check_chroma", _slow), patch.object(health, "_check_database", _ok), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=_resp(200)):
        r = await health.readiness_check()
    assert r.status_code == 503 and b'"chroma":"timeout"' in r.body


@pytest.mark.asyncio
async def test_redis_and_neo4j_only_when_enabled(monkeypatch):
    monkeypatch.setenv("CACHE_BACKEND", "redis")
    monkeypatch.setenv("DISABLE_GRAPH", "0")
    with patch.object(health, "_check_chroma", _ok), patch.object(health, "_check_database", _ok), \
         patch.object(health, "_check_redis", _boom), patch.object(health, "_check_neo4j", _ok), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=_resp(200)):
        r = await health.readiness_check()
    assert r.status_code == 503 and b'"redis":"unavailable"' in r.body and b'"neo4j":"ok"' in r.body


@pytest.mark.asyncio
async def test_llm_check_is_cached_and_informational(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    with patch.object(health, "_check_chroma", _ok), patch.object(health, "_check_database", _ok), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=_resp(401)) as get:
        r1 = await health.readiness_check()
        r2 = await health.readiness_check()
    assert get.await_count == 1  # second probe served from cache
    assert r1.status_code == 200 and r2.status_code == 200  # LLM never fails readiness
    assert b'"llm":"http_401"' in r1.body


@pytest.mark.asyncio
async def test_database_check_uses_store():
    store = MagicMock()
    store.user_exists.return_value = False
    with patch("app.core.auth.get_user_store", return_value=store):
        await health._check_database()
    store.user_exists.assert_called_once_with(0)
