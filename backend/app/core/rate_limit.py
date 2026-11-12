import ipaddress
import os
from functools import lru_cache

from limits import parse
from slowapi import Limiter
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Rate limits configurable via environment variables
DEFAULT_RATE_LIMIT = os.getenv("DEFAULT_RATE_LIMIT", "60/minute")
CHAT_RATE_LIMIT = os.getenv("CHAT_RATE_LIMIT", "5/minute")
UPLOAD_RATE_LIMIT = os.getenv("UPLOAD_RATE_LIMIT", "3/minute")
URL_RATE_LIMIT = os.getenv("URL_RATE_LIMIT", "2/minute")
AUTH_RATE_LIMIT = os.getenv("AUTH_RATE_LIMIT", "5/minute")


@lru_cache(maxsize=8)
def _trusted_networks(raw: str):
    nets = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            nets.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            continue
    return tuple(nets)


def _is_trusted_proxy(peer: str) -> bool:
    nets = _trusted_networks(os.getenv("TRUSTED_PROXY_CIDRS", ""))
    if not nets or not peer:
        return False
    try:
        addr = ipaddress.ip_address(peer)
    except ValueError:
        return False
    return any(addr in net for net in nets)


def client_ip(request: Request) -> str:
    """Client IP, honouring X-Forwarded-For only from trusted proxies (R-06).

    nginx sets `X-Forwarded-For $remote_addr` (overwriting client values), so
    the first hop is the real client when the direct peer is our proxy. Requests
    from anywhere else cannot spoof their key with the header.
    """
    peer = request.client.host if request.client else ""
    if _is_trusted_proxy(peer):
        xff = request.headers.get("x-forwarded-for", "")
        first = xff.split(",")[0].strip()
        if first:
            try:
                ipaddress.ip_address(first)
                return first
            except ValueError:
                pass
    return peer or "unknown"


def ip_key(request: Request) -> str:
    return "ip:" + client_ip(request)


def user_or_ip_key(request: Request) -> str:
    """Key by authenticated user id when a valid JWT is present, else by IP."""
    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        try:
            import jwt

            from app.core.auth import JWT_ALGORITHM, JWT_SECRET

            payload = jwt.decode(auth[7:].strip(), JWT_SECRET, algorithms=[JWT_ALGORITHM])
            sub = payload.get("sub")
            if sub:
                return f"user:{sub}"
        except Exception:
            pass  # invalid/expired tokens fall back to IP; the route will reject them
    return ip_key(request)


def _storage_uri() -> str:
    explicit = os.getenv("RATE_LIMIT_STORAGE_URI", "").strip()
    if explicit:
        return explicit
    if os.getenv("CACHE_BACKEND") == "redis":
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        password = os.getenv("REDIS_PASSWORD")
        if password and "@" not in url:
            url = url.replace("redis://", f"redis://:{password}@")
        return url
    return "memory://"


# Per-route limits use the @limiter.limit decorators. Shared across workers and
# containers when backed by Redis.
limiter = Limiter(
    key_func=user_or_ip_key,
    storage_uri=_storage_uri(),
    # Never take the API down because the limiter storage is unreachable.
    swallow_errors=True,
)


class DefaultRateLimitMiddleware(BaseHTTPMiddleware):
    """Global DEFAULT_RATE_LIMIT ceiling for every request.

    slowapi's SlowAPIMiddleware resolves the endpoint by walking `app.routes`;
    since FastAPI 0.16x wraps included routers in objects without `endpoint`,
    it found no handler and silently exempted every request (found in the
    post-upgrade runtime check). This middleware does not need the endpoint:
    it applies the ceiling to the same user-or-IP key as the per-route limits.
    """

    async def dispatch(self, request: Request, call_next):
        lim = request.app.state.limiter
        if lim.enabled:
            raw = os.getenv("DEFAULT_RATE_LIMIT", DEFAULT_RATE_LIMIT)
            try:
                item = parse(raw)
                allowed = lim.limiter.hit(item, "global", user_or_ip_key(request))
            except Exception:
                allowed = True  # storage outage: fail open, like swallow_errors
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"error": f"Rate limit exceeded: {raw}"},
                    headers={"Retry-After": "60"},
                )
        return await call_next(request)
