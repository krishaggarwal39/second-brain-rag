import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

try:
    import psycopg2
    from psycopg2 import sql
except ImportError:
    psycopg2 = None
    sql = None

JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_EXPIRY_MINUTES = int(os.getenv("JWT_EXPIRY_MINUTES", "1440"))
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
        self._lock = threading.Lock()

    def user_exists(self, user_id: int) -> bool:
        with self._lock:
            cursor = self.conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,))
            return cursor.fetchone() is not None

    def create_user(self, email: str, password: str, role: str = "user") -> int:
        hashed = pwd_context.hash(password)
        with self._lock:
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
            cursor = self.conn.execute("SELECT id, hashed_password FROM users WHERE email = ?", (email,))
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

    def user_exists(self, user_id: int) -> bool:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM users WHERE id = %s", (user_id,))
                return cur.fetchone() is not None

    def create_user(self, email: str, password: str, role: str = "user") -> int:
        hashed = pwd_context.hash(password)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
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
                cur.execute("SELECT id, hashed_password FROM users WHERE email = %s", (email,))
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
    return "admin" if email.strip().lower() in admin_emails else "user"


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
    if not store.user_exists(user_id):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user_id


async def get_current_role(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    payload = _decode_token(credentials)
    return payload.get("role") or "user"


def require_role(*allowed: str):
    async def role_checker(
        user_id: int = Depends(get_current_user),
        role: str = Depends(get_current_role),
    ) -> int:
        allowed_roles = {r.lower() for r in allowed}
        if role.lower() not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user_id

    return role_checker
