import asyncio
import hashlib
import os
import time

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field

from app.core.auth import (
    create_access_token,
    get_user_store,
    normalize_email,
    resolve_role,
    validate_password_strength,
)
from app.core.cache import delete_cache, get_cache, set_cache
from app.core.rate_limit import AUTH_RATE_LIMIT, ip_key, limiter

router = APIRouter(prefix="/auth", tags=["Auth"])

# Per-email failed-login lockout (R-05). Works with the Redis or in-memory cache.
LOGIN_MAX_FAILURES = int(os.getenv("LOGIN_MAX_FAILURES", "5"))
LOGIN_LOCKOUT_SECONDS = int(os.getenv("LOGIN_LOCKOUT_SECONDS", "900"))


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def _fail_key(email: str) -> str:
    # Hash so raw addresses are never stored as cache keys.
    return "authfail:" + hashlib.sha256(normalize_email(email).encode()).hexdigest()


def _env_admin_sync_enabled() -> bool:
    return os.getenv("ALLOW_ADMIN_SELF_REGISTER", "false").strip().lower() in ("1", "true", "yes")


def _sync_env_role(store, user_id: int, email: str) -> str:
    """Return the stored role; optionally sync it from ADMIN_EMAILS (dev only).

    R-04: there is no email verification, so whoever registers an address in
    ADMIN_EMAILS first would become admin. By default roles are therefore only
    changed out of band with `python -m app.cli promote|demote <email>`, and
    the stored role is authoritative for authorization (see require_role).
    ALLOW_ADMIN_SELF_REGISTER=true restores env-based sync for local dev.
    """
    if _env_admin_sync_enabled() and os.getenv("ADMIN_EMAILS", "").strip():
        role = resolve_role(email)
        store.set_role(user_id, role)
        return role
    return store.get_role(user_id)


@router.post("/register", response_model=TokenResponse)
@limiter.limit(AUTH_RATE_LIMIT, key_func=ip_key)
async def register(request: Request, body: RegisterRequest):
    email = normalize_email(body.email)
    try:
        validate_password_strength(body.password, email)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    store = get_user_store()
    try:
        user_id = await asyncio.to_thread(store.create_user, email, body.password)
    except ValueError:
        # create_user hashes the password before its duplicate check, so a
        # duplicate costs the same bcrypt time as a real registration. Existence
        # is still revealed by the 409 status: accepted trade-off for open
        # registration, bounded by AUTH_RATE_LIMIT.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    role = await asyncio.to_thread(_sync_env_role, store, user_id, email)
    return TokenResponse(access_token=create_access_token(user_id, role=role))


@router.post("/login", response_model=TokenResponse)
@limiter.limit(AUTH_RATE_LIMIT, key_func=ip_key)
async def login(request: Request, body: LoginRequest):
    email = normalize_email(body.email)
    key = _fail_key(email)
    state = await get_cache(key) or {}
    locked_until = float(state.get("locked_until", 0))
    now = time.time()
    if locked_until > now:
        retry = int(locked_until - now) + 1
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Too many failed login attempts. Try again later."},
            headers={"Retry-After": str(retry)},
        )

    store = get_user_store()
    user_id = await asyncio.to_thread(store.authenticate, email, body.password)
    if not user_id:
        failures = int(state.get("failures", 0)) + 1
        new_state = {"failures": failures}
        if failures >= LOGIN_MAX_FAILURES:
            new_state = {"failures": 0, "locked_until": now + LOGIN_LOCKOUT_SECONDS}
        await set_cache(key, new_state, ttl=LOGIN_LOCKOUT_SECONDS)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    await delete_cache(key)
    role = await asyncio.to_thread(_sync_env_role, store, user_id, email)
    return TokenResponse(access_token=create_access_token(user_id, role=role))
