import os
import sqlite3
import threading
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from passlib.context import CryptContext
import jwt
from jwt import PyJWTError as JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

try:
    import psycopg2
    from psycopg2 import sql
except ImportError:
    psycopg2 = None
    sql = None

JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES") or os.getenv("JWT_EXPIRY_MINUTES") or "480")
JWT_ALGORITHM = "HS256"

def validate_jwt_secret(secret: str | None) -> None:
    """Validate that the JWT secret is non-empty, >=16 chars, and does not contain 'change-me'."""
    if not secret or len(secret) < 16 or "change-me" in secret.lower():
        raise RuntimeError(
            "JWT_SECRET is not configured or too weak. "
            "Set a strong secret in .env (minimum 16 characters, not containing 'change-me')."
        )

# Validate once at import/startup (module load)
validate_jwt_secret(JWT_SECRET)

_BM25_DATA_DIR = os.getenv("BM25_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "auth_data"))
AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", os.path.join(os.path.abspath(_BM25_DATA_DIR), "users.db"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

# Constant fallback bcrypt hash for timing attack mitigation
_DUMMY_BCRYPT_HASH = "$2b$12$2NK5zg.pDqK6FCWdq0nEher3FLIp032FxthXuJFXDtHvTEUTlUgSq"


def normalize_email(email: str | None) -> str:
    """Canonical form used for storage and lookup (R-04): trimmed and lowercased."""
    return (email or "").strip().lower()


# Small embedded list of very common passwords (R-05). Not exhaustive by design.
_COMMON_PASSWORDS = frozenset({
    "password", "password1", "password12", "password123", "password1234", "passw0rd",
    "12345678", "123456789", "1234567890", "11111111", "00000000", "87654321",
    "qwerty123", "qwertyuiop", "qwerty12", "1q2w3e4r", "1qaz2wsx", "zaq12wsx",
    "abc12345", "abcd1234", "iloveyou", "sunshine", "princess", "football",
    "baseball", "welcome1", "welcome123", "letmein1", "trustno1", "dragon12",
    "monkey123", "superman", "starwars", "whatever", "computer", "internet",
    "admin123", "administrator", "changeme", "changeme1", "secret123", "default1",
    "test1234", "testtest", "asdfghjk", "asdf1234", "michael1", "shadow12",
    "master12", "freedom1", "strongpassword", "strongpassword123",
})


def validate_password_strength(password: str, email: str = "") -> None:
    """Raise ValueError for passwords that are trivially guessable (R-05)."""
    pw = password or ""
    if len(pw) < 8 or len(pw) > 128:
        raise ValueError("Password must be between 8 and 128 characters")
    if pw.lower() in _COMMON_PASSWORDS:
        raise ValueError("Password is too common")
    local = normalize_email(email).split("@")[0]
    if local and pw.lower() == local:
        raise ValueError("Password must not equal the email name")


class UserStore:
    def __init__(self):
        os.makedirs(os.path.dirname(AUTH_DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(AUTH_DB_PATH, check_same_thread=False, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                created_at TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
            )
        """)
        # Guarded migration: ensure role column exists on users
        cursor = self.conn.execute("PRAGMA table_info(users);")
        columns = [row[1] for row in cursor.fetchall()]
        if "role" not in columns:
            self.conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user';")
        self.conn.commit()
        _ensure_ci_unique_sqlite(self.conn)
        self._lock = threading.Lock()

    def user_exists(self, user_id: int) -> bool:
        with self._lock:
            cursor = self.conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,))
            return cursor.fetchone() is not None

    def create_user(self, email: str, password: str, role: str = "user") -> int:
        email = normalize_email(email)
        hashed = pwd_context.hash(password)
        with self._lock:
            # Case-insensitive duplicate check (legacy rows may be mixed case).
            if self.conn.execute(
                "SELECT 1 FROM users WHERE lower(email) = ?", (email,)
            ).fetchone():
                raise ValueError("Email already registered")
            try:
                cursor = self.conn.execute(
                    "INSERT INTO users (email, hashed_password, created_at, role) VALUES (?, ?, ?, ?)",
                    (email, hashed, datetime.now(timezone.utc).isoformat(), role)
                )
                self.conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                raise ValueError("Email already registered")

    def authenticate(self, email: str, password: str):
        with self._lock:
            cursor = self.conn.execute(
                "SELECT id, hashed_password FROM users WHERE lower(email) = ? ORDER BY id LIMIT 1",
                (normalize_email(email),),
            )
            row = cursor.fetchone()
        if not row:
            if hasattr(pwd_context, "dummy_verify"):
                pwd_context.dummy_verify()
            else:
                pwd_context.verify(password, _DUMMY_BCRYPT_HASH)
            return None
        if not pwd_context.verify(password, row[1]):
            return None
        return row[0]

    def get_role(self, user_id: int) -> str:
        with self._lock:
            cursor = self.conn.execute("SELECT role FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return row[0] if row and row[0] else "user"

    def set_role(self, user_id: int, role: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
            self.conn.commit()


    def get_user_id_by_email(self, email: str) -> int | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT id FROM users WHERE lower(email) = ? ORDER BY id LIMIT 1",
                (normalize_email(email),),
            ).fetchone()
        return row[0] if row else None


def _ensure_ci_unique_sqlite(conn) -> None:
    """Add a case-insensitive unique index when legacy data allows it.

    Legacy databases may already hold case-variant duplicates (the R-04 exploit
    created them). We never delete user rows automatically; lookups resolve to
    the oldest row instead, and the index is added once duplicates are gone.
    """
    try:
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_users_email_lower ON users(lower(email))")
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        import logging
        logging.getLogger(__name__).warning(
            "users table has case-variant duplicate emails; case-insensitive unique index not created"
        )


SQLiteUserStore = UserStore


class PostgresUserStore:
    """PostgreSQL user store using short-lived psycopg2 connections per operation.

    Matches the multi-worker concurrency pattern of PostgresBM25Store without
    SQLite single-writer bottleneck.
    """

    def __init__(self, database_url: str | None = None):
        if psycopg2 is None:
            raise ImportError(
                "psycopg2 is required for PostgresUserStore. "
                "Please install psycopg2 or psycopg2-binary."
            )
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL is not set. Please provide database_url or set the "
                "DATABASE_URL environment variable."
            )
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Yield a short-lived PostgreSQL database connection per operation."""
        conn = psycopg2.connect(self.database_url)
        try:
            yield conn
        except Exception:
            if not conn.closed:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            if not conn.closed:
                conn.close()

    def _init_db(self):
        """Idempotently create users table."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        hashed_password TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        role TEXT NOT NULL DEFAULT 'user'
                    );
                """)
                # Guarded migration: ensure role column exists on users
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'users';
                """)
                user_cols = {row[0] for row in cur.fetchall()}
                if user_cols and "role" not in user_cols:
                    cur.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user';")
            conn.commit()
            # Case-insensitive uniqueness (R-04). Skipped if legacy duplicates exist.
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS ux_users_email_lower ON users (lower(email));"
                    )
                conn.commit()
            except psycopg2.Error:
                conn.rollback()

    def user_exists(self, user_id: int) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM users WHERE id = %s", (user_id,))
                return cur.fetchone() is not None

    def create_user(self, email: str, password: str, role: str = "user") -> int:
        email = normalize_email(email)
        hashed = pwd_context.hash(password)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("SELECT 1 FROM users WHERE lower(email) = %s", (email,))
                    if cur.fetchone():
                        raise ValueError("Email already registered")
                    cur.execute(
                        "INSERT INTO users (email, hashed_password, created_at, role) VALUES (%s, %s, %s, %s) RETURNING id;",
                        (email, hashed, datetime.now(timezone.utc).isoformat(), role),
                    )
                    user_id = cur.fetchone()[0]
                    conn.commit()
                    return user_id
                except psycopg2.IntegrityError:
                    conn.rollback()
                    raise ValueError("Email already registered")

    def authenticate(self, email: str, password: str) -> int | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, hashed_password FROM users WHERE lower(email) = %s ORDER BY id LIMIT 1",
                    (normalize_email(email),),
                )
                row = cur.fetchone()
        if not row:
            if hasattr(pwd_context, "dummy_verify"):
                pwd_context.dummy_verify()
            else:
                pwd_context.verify(password, _DUMMY_BCRYPT_HASH)
            return None
        if not pwd_context.verify(password, row[1]):
            return None
        return row[0]

    def get_role(self, user_id: int) -> str:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT role FROM users WHERE id = %s", (user_id,))
                row = cur.fetchone()
                return row[0] if row and row[0] else "user"

    def set_role(self, user_id: int, role: str) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET role = %s WHERE id = %s", (role, user_id))
            conn.commit()


    def get_user_id_by_email(self, email: str) -> int | None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM users WHERE lower(email) = %s ORDER BY id LIMIT 1",
                    (normalize_email(email),),
                )
                row = cur.fetchone()
        return row[0] if row else None


_user_store: UserStore | PostgresUserStore | None = None
_current_backend: str | None = None


def get_user_store() -> UserStore | PostgresUserStore:
    global _user_store, _current_backend
    sparse_backend = os.getenv("SPARSE_BACKEND", "sqlite").lower().strip()
    database_url = os.getenv("DATABASE_URL")
    backend = "postgres" if (sparse_backend == "postgres" and database_url) else "sqlite"

    if _user_store is None or _current_backend != backend:
        if backend == "postgres":
            _user_store = PostgresUserStore(database_url=database_url)
            _current_backend = "postgres"
        else:
            _user_store = UserStore()
            _current_backend = "sqlite"
    return _user_store


def reset_user_store() -> None:
    """Reset cached singleton instance (useful for testing or backend switching)."""
    global _user_store, _current_backend
    _user_store = None
    _current_backend = None


def resolve_role(email: str) -> str:
    """Resolve role for an email address against ADMIN_EMAILS environment variable."""
    admin_emails_raw = os.getenv("ADMIN_EMAILS", "")
    if not admin_emails_raw or not email:
        return "user"
    admin_emails = {e.strip().lower() for e in admin_emails_raw.split(",") if e.strip()}
    return "admin" if normalize_email(email) in admin_emails else "user"


def create_access_token(user_id: int, role: str = "user") -> str:
    validate_jwt_secret(JWT_SECRET)
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRY_MINUTES)
    payload = {"sub": str(user_id), "role": role, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(credentials: HTTPAuthorizationCredentials | None) -> dict:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    try:
        return jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> int:
    payload = _decode_token(credentials)
    try:
        user_id = int(payload["sub"])
    except (KeyError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    store = get_user_store()
    # R-25: Run synchronous user_exists query in a worker thread to avoid blocking the event loop
    user_exists = await asyncio.to_thread(store.user_exists, user_id)
    if not user_exists:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user_id


async def get_current_role(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Role claim from the token (informational only; never used for authorization)."""
    payload = _decode_token(credentials)
    return payload.get("role") or "user"


async def get_db_role(user_id: int = Depends(get_current_user)) -> str:
    """Authoritative role, read from the user store on every request (R-04).

    Promotion and demotion take effect immediately instead of at token expiry.
    """
    store = get_user_store()
    return await asyncio.to_thread(store.get_role, user_id)


def require_role(*allowed: str):
    async def role_checker(
        user_id: int = Depends(get_current_user),
        role: str = Depends(get_db_role),
    ) -> int:
        allowed_roles = {r.lower() for r in allowed}
        if role.lower() not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user_id

    return role_checker
