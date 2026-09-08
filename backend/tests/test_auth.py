import importlib
import os
import pytest
from unittest.mock import patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import app.core.auth as auth
from app.core.auth import (
    get_user_store,
    validate_jwt_secret,
    create_access_token,
    get_current_user,
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

            # create_user success
            mock_cursor.fetchone.return_value = [42]
            uid = pg_store.create_user("test@example.com", "pass123")
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

