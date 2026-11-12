import importlib
import os
import sqlite3
from datetime import datetime, timezone, timedelta
import pytest
from unittest.mock import patch
import jwt
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import app.core.auth as auth
from app.core.auth import (
    get_user_store,
    validate_jwt_secret,
    create_access_token,
    get_current_user,
    get_current_role,
    require_role,
    resolve_role,
)

DATABASE_URL = os.getenv("DATABASE_URL")
requires_postgres = pytest.mark.skipif(
    not DATABASE_URL,
    reason="DATABASE_URL environment variable is not set (PostgreSQL tests skipped)",
)


class TestUserExistsAndTokenInvalidation:
    """FLAW 3: Verify user existence check in UserStore and get_current_user."""

    def test_user_exists_returns_true_for_existing_user(self):
        store = get_user_store()
        email = f"exists_{os.urandom(4).hex()}@example.com"
        user_id = store.create_user(email, "password12345")
        assert store.user_exists(user_id) is True

    def test_user_exists_returns_false_for_nonexistent_user(self):
        store = get_user_store()
        assert store.user_exists(99999999) is False

    @pytest.mark.asyncio
    async def test_get_current_user_succeeds_for_existing_user(self):
        store = get_user_store()
        email = f"valid_{os.urandom(4).hex()}@example.com"
        user_id = store.create_user(email, "password12345")

        token = create_access_token(user_id)
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        result_id = await get_current_user(creds)
        assert result_id == user_id

    @pytest.mark.asyncio
    async def test_get_current_user_raises_401_when_user_missing(self):
        # Mint token for user id that does not exist in store
        token = create_access_token(88888888)
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(creds)
        assert exc_info.value.status_code == 401
        assert "User not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_current_user_missing_credentials_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(None)
        assert exc_info.value.status_code == 401
        assert "Authentication required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token_raises_401(self):
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid.jwt.token")
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(creds)
        assert exc_info.value.status_code == 401
        assert "Invalid or expired token" in exc_info.value.detail


class TestJwtSecretGuard:
    """FLAW 4: Strengthened JWT secret validation at startup and token creation."""

    @pytest.mark.parametrize("weak_secret", [
        "",
        None,
        "short",
        "123456789012345",  # 15 chars (< 16)
        "change-me",
        "change-me-to-a-strong-jwt-secret",
        "CHANGE-ME-TO-A-STRONG-JWT-SECRET",
        "Change-Me-With-Long-String-123456",
        "secret-with-CHANGE-me-in-the-middle",
    ])
    def test_validate_jwt_secret_rejects_weak_secrets(self, weak_secret):
        with pytest.raises(RuntimeError, match="JWT_SECRET is not configured or too weak"):
            validate_jwt_secret(weak_secret)

    @pytest.mark.parametrize("strong_secret", [
        "this-is-a-valid-strong-secret-key",
        "super-random-crypto-key-1234567890",
        "a" * 16,
        "a" * 64,
    ])
    def test_validate_jwt_secret_accepts_strong_secrets(self, strong_secret):
        # Should not raise any exception
        validate_jwt_secret(strong_secret)

    def test_create_access_token_rejects_weak_secret(self, monkeypatch):
        monkeypatch.setattr(auth, "JWT_SECRET", "change-me-to-a-strong-jwt-secret")
        with pytest.raises(RuntimeError, match="JWT_SECRET is not configured or too weak"):
            create_access_token(1)

        monkeypatch.setattr(auth, "JWT_SECRET", "too-short")
        with pytest.raises(RuntimeError, match="JWT_SECRET is not configured or too weak"):
            create_access_token(1)

        monkeypatch.setattr(auth, "JWT_SECRET", "")
        with pytest.raises(RuntimeError, match="JWT_SECRET is not configured or too weak"):
            create_access_token(1)

    def test_startup_guard_fails_on_placeholder_secret(self):
        with patch.dict(os.environ, {"JWT_SECRET": "change-me-to-a-strong-jwt-secret"}):
            with pytest.raises(RuntimeError, match="JWT_SECRET is not configured or too weak"):
                importlib.reload(auth)

        # Restore valid test secret
        with patch.dict(os.environ, {"JWT_SECRET": "test-jwt-secret-key-for-testing-only-12345"}):
            importlib.reload(auth)


class TestTimingAttackMitigation:
    """FLAW 5: Timing attack mitigation in UserStore.authenticate."""

    def test_authenticate_returns_user_id_for_valid_credentials(self):
        store = get_user_store()
        email = f"auth_test_{os.urandom(4).hex()}@example.com"
        password = "validpassword123"
        user_id = store.create_user(email, password)

        auth_id = store.authenticate(email, password)
        assert auth_id == user_id

    def test_authenticate_returns_none_for_wrong_password(self):
        store = get_user_store()
        email = f"auth_test_{os.urandom(4).hex()}@example.com"
        password = "validpassword123"
        store.create_user(email, password)

        result = store.authenticate(email, "wrongpassword456")
        assert result is None

    def test_authenticate_returns_none_for_unknown_email(self):
        store = get_user_store()
        result = store.authenticate("completely_unknown_user@example.com", "anypassword")
        assert result is None

    def test_authenticate_executes_dummy_verification_on_unknown_email(self):
        store = get_user_store()
        unknown_email = f"missing_{os.urandom(6).hex()}@example.com"

        with patch.object(auth.pwd_context, "dummy_verify", wraps=auth.pwd_context.dummy_verify) as mock_dummy:
            result = store.authenticate(unknown_email, "dummy_password")
            assert result is None
            mock_dummy.assert_called_once()


class TestUserStoreBackendSelection:
    """FLAW 6: Verify UserStore backend selection and PostgresUserStore behavior."""

    def test_default_is_sqlite(self, monkeypatch):
        auth.reset_user_store()
        monkeypatch.delenv("SPARSE_BACKEND", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        store = get_user_store()
        assert isinstance(store, auth.UserStore)
        assert isinstance(store, auth.SQLiteUserStore)

    def test_explicit_sqlite_backend(self, monkeypatch):
        auth.reset_user_store()
        monkeypatch.setenv("SPARSE_BACKEND", "sqlite")
        store = get_user_store()
        assert isinstance(store, auth.UserStore)

    def test_postgres_backend_without_database_url_falls_back_to_sqlite(self, monkeypatch):
        auth.reset_user_store()
        monkeypatch.setenv("SPARSE_BACKEND", "postgres")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        store = get_user_store()
        assert isinstance(store, auth.UserStore)

    def test_postgres_backend_with_database_url_selected(self, monkeypatch):
        from unittest.mock import MagicMock
        auth.reset_user_store()
        monkeypatch.setenv("SPARSE_BACKEND", "postgres")
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")

        with patch("app.core.auth.psycopg2.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            store = get_user_store()
            assert isinstance(store, auth.PostgresUserStore)

        auth.reset_user_store()

    def test_postgres_user_store_operations(self, monkeypatch):
        from unittest.mock import MagicMock
        import psycopg2

        fake_url = "postgresql://user:pass@localhost:5432/testdb"
        with patch("app.core.auth.psycopg2.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            # Init
            pg_store = auth.PostgresUserStore(database_url=fake_url)

            # create_user success: first fetchone is the case-insensitive
            # duplicate check (no row), second is RETURNING id.
            mock_cursor.fetchone.side_effect = [None, [42]]
            uid = pg_store.create_user("Test@Example.com", "pass123")
            mock_cursor.fetchone.side_effect = None
            assert uid == 42
            insert_args = mock_cursor.execute.call_args[0][1]
            assert insert_args[0] == "test@example.com"  # stored normalised (R-04)
            assert uid == 42
            assert mock_conn.commit.called

            # create_user duplicate email raises ValueError
            mock_cursor.execute.side_effect = psycopg2.IntegrityError("duplicate key")
            with pytest.raises(ValueError, match="Email already registered"):
                pg_store.create_user("duplicate@example.com", "pass123")
            mock_cursor.execute.side_effect = None

            # user_exists
            mock_cursor.fetchone.return_value = [1]
            assert pg_store.user_exists(42) is True
            mock_cursor.fetchone.return_value = None
            assert pg_store.user_exists(999) is False

            # authenticate success
            hashed = auth.pwd_context.hash("mypassword")
            mock_cursor.fetchone.return_value = (42, hashed)
            assert pg_store.authenticate("test@example.com", "mypassword") == 42

            # authenticate wrong password
            mock_cursor.fetchone.return_value = (42, hashed)
            assert pg_store.authenticate("test@example.com", "wrongpass") is None

            # authenticate unknown email
            mock_cursor.fetchone.return_value = None
            with patch.object(auth.pwd_context, "dummy_verify", wraps=auth.pwd_context.dummy_verify) as mock_dummy:
                assert pg_store.authenticate("missing@example.com", "any") is None
                mock_dummy.assert_called_once()

            # get_role
            mock_cursor.fetchone.return_value = ("admin",)
            assert pg_store.get_role(42) == "admin"
            mock_cursor.fetchone.return_value = None
            assert pg_store.get_role(42) == "user"

            # set_role
            pg_store.set_role(42, "admin")
            mock_cursor.execute.assert_called_with("UPDATE users SET role = %s WHERE id = %s", ("admin", 42))
            assert mock_conn.commit.called


class TestRoleColumnAndStore:
    """Verify role column in UserStore and PostgresUserStore with guarded migrations."""

    def test_sqlite_role_column_exists_and_defaults_to_user(self):
        store = get_user_store()
        email = f"role_test_{os.urandom(4).hex()}@example.com"
        uid = store.create_user(email, "password123")
        assert store.get_role(uid) == "user"

        with store._lock:
            cur = store.conn.execute("SELECT role FROM users WHERE id = ?", (uid,))
            row = cur.fetchone()
            assert row is not None
            assert row[0] == "user"

    def test_sqlite_guarded_migration_adds_missing_role_column(self, tmp_path, monkeypatch):
        test_db = str(tmp_path / "legacy_users.db")
        conn = sqlite3.connect(test_db)
        conn.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "INSERT INTO users (email, hashed_password, created_at) VALUES (?, ?, ?)",
            ("legacy@example.com", "fakehash", "2026-01-01T00:00:00Z"),
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr(auth, "AUTH_DB_PATH", test_db)
        legacy_store = auth.UserStore()
        cursor = legacy_store.conn.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cursor.fetchall()]
        assert "role" in columns
        assert legacy_store.get_role(1) == "user"

    def test_sqlite_set_role(self):
        store = get_user_store()
        email = f"role_set_{os.urandom(4).hex()}@example.com"
        uid = store.create_user(email, "password123")
        assert store.get_role(uid) == "user"

        store.set_role(uid, "admin")
        assert store.get_role(uid) == "admin"

    def test_sqlite_get_role_nonexistent_user(self):
        store = get_user_store()
        assert store.get_role(99999999) == "user"

    @requires_postgres
    def test_postgres_role_column_live(self):
        pg_store = auth.PostgresUserStore(database_url=DATABASE_URL)
        email = f"pg_role_{os.urandom(4).hex()}@example.com"
        uid = pg_store.create_user(email, "password123")
        try:
            assert pg_store.get_role(uid) == "user"
            pg_store.set_role(uid, "admin")
            assert pg_store.get_role(uid) == "admin"
        finally:
            with pg_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM users WHERE id = %s", (uid,))
                conn.commit()


class TestResolveRole:
    """Verify resolve_role reads ADMIN_EMAILS env var correctly."""

    def test_resolve_role_returns_admin_for_listed_emails(self, monkeypatch):
        monkeypatch.setenv("ADMIN_EMAILS", "admin@example.com, super@test.com, ROOT@MYDOMAIN.ORG")
        assert resolve_role("admin@example.com") == "admin"
        assert resolve_role("ADMIN@EXAMPLE.COM") == "admin"
        assert resolve_role(" super@test.com ") == "admin"
        assert resolve_role("root@mydomain.org") == "admin"

    def test_resolve_role_returns_user_for_unlisted_or_empty(self, monkeypatch):
        monkeypatch.setenv("ADMIN_EMAILS", "admin@example.com")
        assert resolve_role("other@example.com") == "user"
        assert resolve_role("") == "user"

    def test_resolve_role_defaults_to_user_when_unset(self, monkeypatch):
        monkeypatch.delenv("ADMIN_EMAILS", raising=False)
        assert resolve_role("admin@example.com") == "user"


class TestJwtRoleClaimAndBackwardCompat:
    """Verify role claim in JWT, token creation, and backward compatibility."""

    def test_create_access_token_default_and_custom_role(self):
        token_default = create_access_token(1)
        payload_default = jwt.decode(token_default, auth.JWT_SECRET, algorithms=[auth.JWT_ALGORITHM])
        assert payload_default["role"] == "user"

        token_admin = create_access_token(1, role="admin")
        payload_admin = jwt.decode(token_admin, auth.JWT_SECRET, algorithms=[auth.JWT_ALGORITHM])
        assert payload_admin["role"] == "admin"

    @pytest.mark.asyncio
    async def test_old_style_token_without_role_claim_resolves_to_user(self):
        store = get_user_store()
        email = f"compat_{os.urandom(4).hex()}@example.com"
        uid = store.create_user(email, "password123")

        # Mint token manually with NO 'role' claim in payload
        expire = datetime.now(timezone.utc) + timedelta(minutes=60)
        token = jwt.encode({"sub": str(uid), "exp": expire}, auth.JWT_SECRET, algorithm=auth.JWT_ALGORITHM)
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        assert await get_current_role(creds) == "user"
        assert await get_current_user(creds) == uid

    @pytest.mark.asyncio
    async def test_get_current_role_with_role_claim(self):
        token = create_access_token(42, role="admin")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        assert await get_current_role(creds) == "admin"

    @pytest.mark.asyncio
    async def test_get_current_role_missing_creds_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            await get_current_role(None)
        assert exc_info.value.status_code == 401
        assert "Authentication required" in exc_info.value.detail


class TestRequireRoleDependency:
    """Verify require_role dependency enforcement."""

    @pytest.mark.asyncio
    async def test_require_role_allows_authorized_role(self):
        checker = require_role("admin")
        res = await checker(user_id=10, role="admin")
        assert res == 10

    @pytest.mark.asyncio
    async def test_require_role_blocks_unauthorized_role(self):
        checker = require_role("admin")
        with pytest.raises(HTTPException) as exc_info:
            await checker(user_id=10, role="user")
        assert exc_info.value.status_code == 403
        assert "Insufficient permissions" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_require_role_multiple_allowed_roles(self):
        checker = require_role("admin", "editor")
        assert await checker(user_id=10, role="editor") == 10
        with pytest.raises(HTTPException) as exc_info:
            await checker(user_id=10, role="viewer")
        assert exc_info.value.status_code == 403

