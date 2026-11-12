"""Batch 4 hardening: R-04, R-05, R-06, R-36, R-37, R-45."""
import os
import uuid

import pytest
from fastapi.testclient import TestClient
import jwt
from starlette.requests import Request

import app.core.auth as auth_mod
from app.core.auth import get_user_store, normalize_email, validate_password_strength

PW = "Corr3ct-Horse-Battery"


@pytest.fixture
def client():
    from app.main import app
    from app.core.rate_limit import limiter

    limiter.reset()
    with TestClient(app) as c:
        yield c
    limiter.reset()


def _role_of(token: str) -> str:
    return jwt.decode(token, auth_mod.JWT_SECRET, algorithms=[auth_mod.JWT_ALGORITHM])["role"]


def _uid(token: str) -> int:
    return int(jwt.decode(token, auth_mod.JWT_SECRET, algorithms=[auth_mod.JWT_ALGORITHM])["sub"])


# ---------------------------------------------------------------- R-04
class TestAdminRoleHardening:
    def test_case_variant_cannot_claim_admin(self, client, monkeypatch):
        """Runtime-confirmed exploit: Admin@Example.com became admin."""
        base = f"admin_{uuid.uuid4().hex[:8]}@example.com"
        monkeypatch.setenv("ADMIN_EMAILS", base)
        monkeypatch.delenv("ALLOW_ADMIN_SELF_REGISTER", raising=False)

        r = client.post("/auth/register", json={"email": base.upper(), "password": PW})
        assert r.status_code == 200
        assert _role_of(r.json()["access_token"]) == "user"

    def test_case_variant_is_duplicate(self, client):
        email = f"dup_{uuid.uuid4().hex[:8]}@example.com"
        assert client.post("/auth/register", json={"email": email, "password": PW}).status_code == 200
        r = client.post("/auth/register", json={"email": email.upper(), "password": PW})
        assert r.status_code == 409

    def test_login_is_case_insensitive(self, client):
        email = f"ci_{uuid.uuid4().hex[:8]}@example.com"
        client.post("/auth/register", json={"email": email, "password": PW})
        r = client.post("/auth/login", json={"email": "  " + email.upper(), "password": PW})
        # EmailStr strips nothing, but our normalisation must; pydantic may reject spaces
        if r.status_code == 422:
            r = client.post("/auth/login", json={"email": email.upper(), "password": PW})
        assert r.status_code == 200

    def test_metrics_uses_db_role_not_token_claim(self, client, monkeypatch):
        api_key = os.getenv("API_KEY", "test-api-key-12345")
        email = f"forge_{uuid.uuid4().hex[:8]}@example.com"
        r = client.post("/auth/register", json={"email": email, "password": PW})
        uid = _uid(r.json()["access_token"])
        forged = auth_mod.create_access_token(uid, role="admin")  # stale/forged claim
        headers = {"X-API-Key": api_key, "Authorization": f"Bearer {forged}"}
        assert client.get("/metrics", headers=headers).status_code == 403

        get_user_store().set_role(uid, "admin")  # promotion applies immediately
        assert client.get("/metrics", headers=headers).status_code == 200
        get_user_store().set_role(uid, "user")  # demotion applies immediately
        assert client.get("/metrics", headers=headers).status_code == 403

    def test_cli_promote_and_demote(self, client):
        from app.cli import main, promote_admins_from_env

        email = f"cli_{uuid.uuid4().hex[:8]}@example.com"
        r = client.post("/auth/register", json={"email": email, "password": PW})
        uid = _uid(r.json()["access_token"])
        assert main(["promote", email.upper()]) == 0
        assert get_user_store().get_role(uid) == "admin"
        assert main(["demote", email]) == 0
        assert get_user_store().get_role(uid) == "user"
        assert main(["promote", "nobody-" + email]) == 1

        os.environ["ADMIN_EMAILS"] = email
        try:
            assert promote_admins_from_env() == [email]
        finally:
            os.environ.pop("ADMIN_EMAILS")

    def test_login_does_not_reset_cli_role_when_env_sync_disabled(self, client, monkeypatch):
        monkeypatch.delenv("ALLOW_ADMIN_SELF_REGISTER", raising=False)
        email = f"keep_{uuid.uuid4().hex[:8]}@example.com"
        r = client.post("/auth/register", json={"email": email, "password": PW})
        uid = _uid(r.json()["access_token"])
        get_user_store().set_role(uid, "admin")
        r = client.post("/auth/login", json={"email": email, "password": PW})
        assert _role_of(r.json()["access_token"]) == "admin"

    def test_legacy_mixed_case_rows_resolve_to_oldest(self, tmp_path, monkeypatch):
        import sqlite3

        db = tmp_path / "users.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,"
            " hashed_password TEXT NOT NULL, created_at TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user')"
        )
        h = auth_mod.pwd_context.hash(PW)
        conn.execute("INSERT INTO users (email, hashed_password, created_at) VALUES ('a@x.com', ?, 't')", (h,))
        conn.execute("INSERT INTO users (email, hashed_password, created_at) VALUES ('A@x.com', ?, 't')", (h,))
        conn.commit()
        conn.close()
        monkeypatch.setattr(auth_mod, "AUTH_DB_PATH", str(db))
        store = auth_mod.UserStore()  # must not crash on duplicates, must not delete rows
        assert store.authenticate("A@X.COM", PW) == 1
        assert store.conn.execute("SELECT count(*) FROM users").fetchone()[0] == 2
        with pytest.raises(ValueError):
            store.create_user("a@X.com", PW)


def test_normalize_email():
    assert normalize_email("  Foo@Bar.COM ") == "foo@bar.com"
    assert normalize_email(None) == ""


def test_token_lifetime_default_is_8h():
    assert auth_mod.JWT_EXPIRY_MINUTES in (480, int(os.getenv("JWT_EXPIRY_MINUTES", "480")))


# ---------------------------------------------------------------- R-05
class TestPasswordPolicyAndThrottling:
    @pytest.mark.parametrize("pw", ["password123", "12345678", "Qwerty123".lower(), "strongpassword123"])
    def test_common_passwords_rejected(self, pw):
        with pytest.raises(ValueError):
            validate_password_strength(pw, "x@example.com")

    def test_password_equal_to_email_name_rejected(self):
        with pytest.raises(ValueError):
            validate_password_strength("johnsmith", "JohnSmith@example.com")

    def test_register_rejects_common_password(self, client):
        r = client.post("/auth/register", json={"email": f"w_{uuid.uuid4().hex[:6]}@ex.com", "password": "password123"})
        assert r.status_code == 422

    def test_login_lockout_after_failures(self, client, monkeypatch):
        import app.api.routes.auth as routes

        monkeypatch.setattr(routes, "LOGIN_MAX_FAILURES", 3)
        email = f"lock_{uuid.uuid4().hex[:8]}@example.com"
        client.post("/auth/register", json={"email": email, "password": PW})
        from app.core.rate_limit import limiter

        for _ in range(3):
            limiter.reset()
            assert client.post("/auth/login", json={"email": email, "password": "wrong-pass"}).status_code == 401
        limiter.reset()
        r = client.post("/auth/login", json={"email": email.upper(), "password": PW})
        assert r.status_code == 429
        assert int(r.headers["Retry-After"]) > 0

    def test_login_success_resets_failures(self, client, monkeypatch):
        import app.api.routes.auth as routes
        from app.core.rate_limit import limiter

        monkeypatch.setattr(routes, "LOGIN_MAX_FAILURES", 3)
        email = f"reset_{uuid.uuid4().hex[:8]}@example.com"
        client.post("/auth/register", json={"email": email, "password": PW})
        for _ in range(2):
            limiter.reset()
            client.post("/auth/login", json={"email": email, "password": "wrong-pass"})
        limiter.reset()
        assert client.post("/auth/login", json={"email": email, "password": PW}).status_code == 200
        for _ in range(2):
            limiter.reset()
            assert client.post("/auth/login", json={"email": email, "password": "wrong-pass"}).status_code == 401

    def test_auth_endpoints_ip_rate_limited(self, client):
        codes = [
            client.post("/auth/login", json={"email": "nobody@example.com", "password": "x"}).status_code
            for _ in range(8)
        ]
        assert 429 in codes


# ---------------------------------------------------------------- R-06
def _req(peer: str, headers: dict | None = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "headers": raw, "client": (peer, 1234), "method": "GET", "path": "/"})


class TestRateLimitKeys:
    def test_users_get_independent_keys(self):
        from app.core.rate_limit import user_or_ip_key

        t1 = auth_mod.create_access_token(1)
        t2 = auth_mod.create_access_token(2)
        k1 = user_or_ip_key(_req("10.0.0.5", {"Authorization": f"Bearer {t1}"}))
        k2 = user_or_ip_key(_req("10.0.0.5", {"Authorization": f"Bearer {t2}"}))
        assert k1 == "user:1" and k2 == "user:2"

    def test_invalid_token_falls_back_to_ip(self):
        from app.core.rate_limit import user_or_ip_key

        assert user_or_ip_key(_req("10.0.0.5", {"Authorization": "Bearer garbage"})) == "ip:10.0.0.5"

    def test_xff_ignored_from_untrusted_peer(self, monkeypatch):
        from app.core.rate_limit import client_ip

        monkeypatch.setenv("TRUSTED_PROXY_CIDRS", "")
        assert client_ip(_req("10.0.0.5", {"X-Forwarded-For": "203.0.113.7"})) == "10.0.0.5"
        monkeypatch.setenv("TRUSTED_PROXY_CIDRS", "172.16.0.0/12")
        assert client_ip(_req("10.0.0.5", {"X-Forwarded-For": "203.0.113.7"})) == "10.0.0.5"

    def test_xff_honoured_from_trusted_peer(self, monkeypatch):
        from app.core.rate_limit import client_ip

        monkeypatch.setenv("TRUSTED_PROXY_CIDRS", "172.16.0.0/12")
        assert client_ip(_req("172.18.0.3", {"X-Forwarded-For": "203.0.113.7, 172.18.0.3"})) == "203.0.113.7"
        assert client_ip(_req("172.18.0.3", {"X-Forwarded-For": "not-an-ip"})) == "172.18.0.3"

    def test_storage_uri_derived_from_redis(self, monkeypatch):
        from app.core import rate_limit

        monkeypatch.delenv("RATE_LIMIT_STORAGE_URI", raising=False)
        monkeypatch.setenv("CACHE_BACKEND", "redis")
        monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")
        monkeypatch.setenv("REDIS_PASSWORD", "pw")
        assert rate_limit._storage_uri() == "redis://:pw@redis:6379/0"
        monkeypatch.setenv("CACHE_BACKEND", "memory")
        assert rate_limit._storage_uri() == "memory://"


# ---------------------------------------------------------------- R-36 / R-37
class TestErrorsAndLogging:
    def test_unhandled_exception_is_generic(self):
        from app.main import app

        @app.get("/__boom_test")
        async def boom():
            class Weird(Exception):
                status_code = 418
                detail = "internal secret detail"
            raise Weird("db password=hunter2")

        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/__boom_test")
        assert r.status_code == 500
        assert "secret" not in r.text and "hunter2" not in r.text
        assert r.headers.get("X-Request-ID")

    def test_inbound_request_id_validated(self, client):
        good = "abcDEF12-34_xy"
        assert client.get("/health", headers={"X-Request-ID": good}).headers["X-Request-ID"] == good
        bad = "<script>alert(1)</script>"
        assert client.get("/health", headers={"X-Request-ID": bad}).headers["X-Request-ID"] != bad

    @pytest.mark.parametrize(
        "raw,leak",
        [
            ("https://x.com/v1?key=AIzaSECRET&q=1", "AIzaSECRET"),
            ("Authorization: Bearer eyJhbGciOi.abc.def", "eyJhbGciOi.abc.def"),
            ("redis://:supersecret@redis:6379/0", "supersecret"),
            ("postgresql://user:pgpass@db/x", "pgpass"),
            ("failed with password=hunter2 for user", "hunter2"),
            ("x-api-key: k-123456", "k-123456"),
        ],
    )
    def test_sanitize_error_msg(self, raw, leak):
        from app.core.logging import sanitize_error_msg

        assert leak not in sanitize_error_msg(raw)

    def test_redact_processor_keys_and_nested(self):
        from app.core.logging import redact_processor

        out = redact_processor(None, "info", {
            "event": "x", "password": "p", "X-API-Key": "k",
            "headers": {"Authorization": "Bearer abc"}, "url": "redis://:pw@h/0",
        })
        assert out["password"] == "[REDACTED]" and out["X-API-Key"] == "[REDACTED]"
        assert out["headers"]["Authorization"] == "[REDACTED]"
        assert "pw@" not in out["url"]


class TestGlobalDefaultLimit:
    """Regression: after the FastAPI upgrade slowapi's middleware exempted every route."""

    def test_default_limit_applies_and_is_per_user(self, client, monkeypatch):
        from app.core.rate_limit import limiter

        monkeypatch.setenv("DEFAULT_RATE_LIMIT", "5/minute")
        limiter.reset()
        api_key = os.getenv("API_KEY", "test-api-key-12345")
        a = {"X-API-Key": api_key, "Authorization": f"Bearer {auth_mod.create_access_token(9001)}"}
        b = {"X-API-Key": api_key, "Authorization": f"Bearer {auth_mod.create_access_token(9002)}"}
        codes_a = [client.get("/health", headers=a).status_code for _ in range(7)]
        assert codes_a[:5] == [200] * 5 and codes_a[5:] == [429, 429]
        assert client.get("/health", headers=b).status_code == 200  # independent bucket
        limiter.reset()

    def test_default_limit_unauthenticated_by_ip(self, client, monkeypatch):
        from app.core.rate_limit import limiter

        monkeypatch.setenv("DEFAULT_RATE_LIMIT", "3/minute")
        limiter.reset()
        codes = [client.get("/health").status_code for _ in range(5)]
        assert codes.count(429) == 2
        limiter.reset()
