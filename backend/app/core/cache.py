from datetime import datetime, timezone
import os
import json
import structlog
import time
from typing import Any, Optional

logger = structlog.get_logger(__name__)

_redis_client = None
_local_cache = {}
MAX_LOCAL_KEYS = 10000

async def init_cache():
    global _redis_client
    # R-03: Ensure init_cache is idempotent and does not leak clients if called twice.
    if _redis_client is not None:
        try:
            await close_cache()
        except Exception:
            _redis_client = None

    if os.getenv("CACHE_BACKEND") == "redis":
        try:
            import redis.asyncio as aioredis
        except Exception as e:
            logger.error("Failed to import redis.asyncio", error=str(e))
            _redis_client = None
            raise RuntimeError(f"Failed to import redis.asyncio: {e}") from e

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_password = os.getenv("REDIS_PASSWORD")

        if redis_password and "@" not in redis_url:
            redis_url = redis_url.replace("redis://", f"redis://:{redis_password}@")

        try:
            _redis_client = aioredis.from_url(redis_url, decode_responses=True)
            await _redis_client.ping()
            logger.info("Redis cache initialized")
        except Exception as e:
            # Never log or raise with the URL: it can embed the Redis password.
            logger.error("Failed to connect to Redis", error=type(e).__name__)
            _redis_client = None
            raise RuntimeError(
                f"Could not connect to Redis ({type(e).__name__}); check REDIS_URL and REDIS_PASSWORD"
            ) from None
    else:
        _redis_client = None

async def get_cache(key: str) -> Optional[Any]:
    if _redis_client:
        try:
            val = await _redis_client.get(key)
            return json.loads(val) if val else None
        except Exception as e:
            logger.error("Redis get error", error=str(e), key=key)
            return None
    else:
        item = _local_cache.get(key)
        if item is None:
            return None
        value, expiry = item
        if expiry is not None and time.time() > expiry:
            _local_cache.pop(key, None)
            return None
        return value

async def set_cache(key: str, value: Any, ttl: Optional[int] = 86400):
    if _redis_client:
        try:
            if ttl is None:
                await _redis_client.set(key, json.dumps(value))
            else:
                await _redis_client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            logger.error("Redis set error", error=str(e), key=key)
    else:
        if len(_local_cache) >= MAX_LOCAL_KEYS:
            now = time.time()
            expired_keys = [k for k, v in _local_cache.items() if v[1] is not None and v[1] < now]
            for k in expired_keys:
                _local_cache.pop(k, None)
            
            if len(_local_cache) >= MAX_LOCAL_KEYS:
                keys_to_remove = list(_local_cache.keys())[:1000]
                for k in keys_to_remove:
                    _local_cache.pop(k, None)
                    
        expiry = time.time() + ttl if ttl else None
        _local_cache[key] = (value, expiry)

async def delete_cache(key: str):
    if _redis_client:
        try:
            await _redis_client.delete(key)
        except Exception as e:
            logger.error("Redis delete error", error=str(e), key=key)
    else:
        _local_cache.pop(key, None)

def daily_key(name: str, now: Optional[float] = None) -> str:
    """Metric key for the current UTC day, e.g. queries_today:2026-11-12 (R-39).

    A new key starts at UTC midnight, so "today" counters reset by themselves.
    """
    day = datetime.fromtimestamp(time.time() if now is None else now, tz=timezone.utc)
    return f"{name}:{day.strftime('%Y-%m-%d')}"


async def increment_metric(key: str, amount: float = 1.0, ttl: Optional[int] = None):
    if _redis_client:
        try:
            if isinstance(amount, float) and amount != int(amount):
                await _redis_client.incrbyfloat(key, amount)
            else:
                await _redis_client.incrby(key, int(amount))
            if ttl:
                await _redis_client.expire(key, ttl)
        except Exception as e:
            logger.error("Redis increment error", error=str(e), key=key)
    else:
        item = _local_cache.get(key)
        if item and (item[1] is None or item[1] > time.time()):
            val, expiry = item
        else:
            val, expiry = 0, (time.time() + ttl if ttl else None)
        _local_cache[key] = (val + amount, expiry)

async def get_metric(key: str) -> float:
    if _redis_client:
        try:
            val = await _redis_client.get(key)
            return float(val) if val else 0.0
        except Exception as e:
            logger.error("Redis get metric error", error=str(e), key=key)
            return 0.0
    else:
        item = _local_cache.get(key)
        if item:
            val, expiry = item
            if expiry is not None and time.time() > expiry:
                _local_cache.pop(key, None)
                return 0.0
            return float(val)
        return 0.0


async def bump_cache_generation(owner_id: Optional[str]) -> int:
    """Bump the cache generation counter for owner_id to invalidate cached queries."""
    if not owner_id:
        return 0
    key = f"cache_gen:{owner_id}"
    await increment_metric(key, 1)
    val = await get_metric(key)
    return int(val)

async def get_cache_generation(owner_id: Optional[str]) -> int:
    """Retrieve the current cache generation counter for owner_id (defaults to 0)."""
    if not owner_id:
        return 0
    key = f"cache_gen:{owner_id}"
    val = await get_metric(key)
    return int(val)

async def close_cache():
    global _redis_client
    if _redis_client:
        try:
            if hasattr(_redis_client, "aclose"):
                await _redis_client.aclose()
            elif hasattr(_redis_client, "close"):
                await _redis_client.close()
        except Exception as e:
            logger.warning("Error closing Redis client", error=str(e))
        finally:
            _redis_client = None

