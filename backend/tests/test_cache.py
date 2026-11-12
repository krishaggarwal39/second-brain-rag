"""Tests for the caching layer (in-memory fallback mode)."""

import pytest
import time
from unittest.mock import patch

# Force in-memory mode for testing
with patch.dict("os.environ", {"CACHE_BACKEND": "memory"}):
    import app.core.cache as cache_module
    from app.core.cache import (
        get_cache,
        set_cache,
        delete_cache,
        increment_metric,
        get_metric,
        bump_cache_generation,
        get_cache_generation,
        init_cache,
        close_cache,
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


class FakeRedis:
    """In-memory stub implementing the redis.asyncio client interface."""
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

    async def incrby(self, key: str, amount: int):
        val = int(self.store.get(key, 0)) + amount
        self.store[key] = str(val)
        return val

    async def incrbyfloat(self, key: str, amount: float):
        val = float(self.store.get(key, 0.0)) + amount
        self.store[key] = str(val)
        return val

    async def ping(self):
        return True

    async def aclose(self):
        self.closed = True

    async def close(self):
        self.closed = True


class TestRedisCache:
    @pytest.fixture(autouse=True)
    def setup_fake_redis(self):
        fake = FakeRedis()
        cache_module._redis_client = fake
        yield fake
        cache_module._redis_client = None

    @pytest.mark.asyncio
    async def test_redis_set_and_get(self, setup_fake_redis):
        await set_cache("key1", {"message": "hello"})
        val = await get_cache("key1")
        assert val == {"message": "hello"}

    @pytest.mark.asyncio
    async def test_redis_get_nonexistent(self, setup_fake_redis):
        assert await get_cache("nonexistent") is None

    @pytest.mark.asyncio
    async def test_redis_delete(self, setup_fake_redis):
        await set_cache("to_del", "val")
        assert await get_cache("to_del") == "val"
        await delete_cache("to_del")
        assert await get_cache("to_del") is None

    @pytest.mark.asyncio
    async def test_redis_metrics(self, setup_fake_redis):
        await increment_metric("counter", 1)
        await increment_metric("counter", 2)
        assert await get_metric("counter") == 3.0

    @pytest.mark.asyncio
    async def test_redis_float_metric(self, setup_fake_redis):
        await increment_metric("float_counter", 1.5)
        await increment_metric("float_counter", 2.25)
        assert abs(await get_metric("float_counter") - 3.75) < 0.001

    @pytest.mark.asyncio
    async def test_redis_cache_generation(self, setup_fake_redis):
        assert await get_cache_generation("u1") == 0
        gen = await bump_cache_generation("u1")
        assert gen == 1
        assert await get_cache_generation("u1") == 1


class TestCacheStartup:
    @pytest.mark.asyncio
    async def test_init_cache_memory_mode(self):
        with patch.dict("os.environ", {"CACHE_BACKEND": "memory"}):
            await init_cache()
            assert cache_module._redis_client is None

    @pytest.mark.asyncio
    async def test_init_cache_redis_unreachable_raises(self):
        with patch.dict("os.environ", {
            "CACHE_BACKEND": "redis",
            "REDIS_URL": "redis://127.0.0.1:1/0"
        }):
            with pytest.raises(RuntimeError) as exc_info:
                await init_cache()
            assert "Could not connect to Redis" in str(exc_info.value)
            assert cache_module._redis_client is None

    @pytest.mark.asyncio
    async def test_init_cache_redis_import_error_raises(self):
        with patch.dict("os.environ", {"CACHE_BACKEND": "redis"}):
            with patch.dict("sys.modules", {"redis.asyncio": None}):
                with pytest.raises(RuntimeError) as exc_info:
                    await init_cache()
                assert "Failed to import redis.asyncio" in str(exc_info.value)
                assert cache_module._redis_client is None

    @pytest.mark.asyncio
    async def test_init_cache_redis_success(self):
        fake = FakeRedis()
        with patch.dict("os.environ", {"CACHE_BACKEND": "redis"}), \
             patch("redis.asyncio.from_url", return_value=fake):
            await init_cache()
            assert cache_module._redis_client is fake
            await close_cache()
            assert cache_module._redis_client is None
            assert fake.closed is True

