"""Tests for the caching layer (in-memory fallback mode)."""

import pytest
import time
from unittest.mock import patch

# Force in-memory mode for testing
with patch.dict("os.environ", {"CACHE_BACKEND": "memory"}):
    from app.core.cache import (
        get_cache,
        set_cache,
        delete_cache,
        increment_metric,
        get_metric,
        bump_cache_generation,
        get_cache_generation,
        _local_cache,
    )


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the local cache before each test."""
    _local_cache.clear()
    yield
    _local_cache.clear()


class TestInMemoryCache:
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        await set_cache("test_key", {"value": 42})
        result = await get_cache("test_key")
        assert result == {"value": 42}

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self):
        result = await get_cache("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self):
        await set_cache("to_delete", "hello")
        await delete_cache("to_delete")
        result = await get_cache("to_delete")
        assert result is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        await set_cache("expiring", "data", ttl=1)
        result = await get_cache("expiring")
        assert result == "data"

        # Simulate time passing
        _local_cache["expiring"] = ("data", time.time() - 1)
        result = await get_cache("expiring")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_ttl_persists(self):
        await set_cache("forever", "persistent", ttl=None)
        result = await get_cache("forever")
        assert result == "persistent"

    @pytest.mark.asyncio
    async def test_overwrite_value(self):
        await set_cache("key", "first")
        await set_cache("key", "second")
        result = await get_cache("key")
        assert result == "second"

    @pytest.mark.asyncio
    async def test_stores_complex_types(self):
        data = {"list": [1, 2, 3], "nested": {"a": True}}
        await set_cache("complex", data)
        result = await get_cache("complex")
        assert result == data


class TestMetrics:
    @pytest.mark.asyncio
    async def test_increment_metric(self):
        await increment_metric("test_counter", 1)
        await increment_metric("test_counter", 1)
        result = await get_metric("test_counter")
        assert result == 2.0

    @pytest.mark.asyncio
    async def test_increment_float_metric(self):
        await increment_metric("cost", 0.5)
        await increment_metric("cost", 0.3)
        result = await get_metric("cost")
        assert abs(result - 0.8) < 0.001

    @pytest.mark.asyncio
    async def test_get_nonexistent_metric(self):
        result = await get_metric("never_set")
        assert result == 0.0


class TestCacheGeneration:
    @pytest.mark.asyncio
    async def test_generation_lifecycle_and_isolation(self):
        # Starts at 0
        assert await get_cache_generation("user_1") == 0
        assert await get_cache_generation(None) == 0

        # None does not bump or create keys
        assert await bump_cache_generation(None) == 0
        assert await get_cache_generation(None) == 0

        # Bumping user_1 increments to 1
        gen1 = await bump_cache_generation("user_1")
        assert gen1 == 1
        assert await get_cache_generation("user_1") == 1

        # Bumping user_1 again increments to 2
        gen2 = await bump_cache_generation("user_1")
        assert gen2 == 2
        assert await get_cache_generation("user_1") == 2

        # user_2 is isolated and still at 0
        assert await get_cache_generation("user_2") == 0
        gen_u2 = await bump_cache_generation("user_2")
        assert gen_u2 == 1
        assert await get_cache_generation("user_2") == 1

        # user_1 is still 2
        assert await get_cache_generation("user_1") == 2

