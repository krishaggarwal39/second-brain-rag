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
    if os.getenv("CACHE_BACKEND") == "redis":
        try:
            import aioredis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            redis_password = os.getenv("REDIS_PASSWORD")
            
            if redis_password and "@" not in redis_url:
                redis_url = redis_url.replace("redis://", f"redis://:{redis_password}@")
                
            _redis_client = await aioredis.from_url(redis_url, decode_responses=True)
            await _redis_client.ping()
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning("Redis unavailable, falling back to in-memory dict", error=str(e))
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

async def increment_metric(key: str, amount: float = 1.0):
    if _redis_client:
        try:
            if isinstance(amount, float) and amount != int(amount):
                await _redis_client.incrbyfloat(key, amount)
            else:
                await _redis_client.incrby(key, int(amount))
        except Exception as e:
            logger.error("Redis increment error", error=str(e), key=key)
    else:
        item = _local_cache.get(key)
        if item:
            val, expiry = item
        else:
            val, expiry = 0, None
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
    if _redis_client:
        await _redis_client.close()

