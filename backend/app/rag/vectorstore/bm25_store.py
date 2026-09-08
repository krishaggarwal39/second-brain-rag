import sqlite3
import os
import json
import threading
from contextlib import contextmanager
from typing import List, Dict, Optional, Union
from app.core.logging import get_logger

try:
    import psycopg2
    from psycopg2.extras import execute_batch
    from psycopg2.pool import ThreadedConnectionPool
except ImportError:
    psycopg2 = None
    execute_batch = None
    ThreadedConnectionPool = None

logger = get_logger(__name__)

# Resolve to an absolute path so it works regardless of CWD.
_data_dir = os.getenv("BM25_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "bm25_data"))
BM25_FILE_PATH = os.path.join(os.path.abspath(_data_dir), "bm25_corpus.json")
SQLITE_DB_PATH = os.path.join(os.path.abspath(_data_dir), "bm25.db")

class BM25Store:
    def __init__(self):
        os.makedirs(_data_dir, exist_ok=True)
        self.conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        
        # Initialize schema
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS corpus USING fts5(id UNINDEXED, text, doc_id UNINDEXED, owner_id UNINDEXED);
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS corpus_mapping (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE,
                doc_id TEXT
            );
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS document_metadata (
                doc_id TEXT PRIMARY KEY,
                filename TEXT,
                chunk_count INTEGER,
                status TEXT,
                created_at TEXT,
                owner_id TEXT
            );
        """)
        # Guarded migration: ensure owner_id column exists on document_metadata
        cursor = self.conn.execute("PRAGMA table_info(document_metadata);")
        columns = [row[1] for row in cursor.fetchall()]
        if "owner_id" not in columns:
            self.conn.execute("ALTER TABLE document_metadata ADD COLUMN owner_id TEXT;")
            self.conn.commit()

        # Guarded migration: ensure owner_id column exists on corpus
        cursor = self.conn.execute("PRAGMA table_info(corpus);")
        corpus_columns = [row[1] for row in cursor.fetchall()]
        if "owner_id" not in corpus_columns:
            logger.info("Migrating SQLite FTS5 corpus table to include owner_id...")
            self.conn.execute("DROP TABLE IF EXISTS corpus_new;")
            self.conn.execute(
                "CREATE VIRTUAL TABLE corpus_new USING fts5(id UNINDEXED, text, doc_id UNINDEXED, owner_id UNINDEXED);"
            )
            self.conn.execute("""
                INSERT INTO corpus_new (rowid, id, text, doc_id, owner_id)
                SELECT c.rowid, c.id, c.text, c.doc_id, coalesce(m.owner_id, '')
                FROM corpus c
                LEFT JOIN document_metadata m ON c.doc_id = m.doc_id;
            """)
            self.conn.execute("DROP TABLE corpus;")
            self.conn.execute("ALTER TABLE corpus_new RENAME TO corpus;")
            self.conn.commit()
            logger.info("SQLite FTS5 corpus migration completed.")

        # Create indexes for fast lookup
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mapping_chunk_id ON corpus_mapping(chunk_id);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mapping_doc_id ON corpus_mapping(doc_id);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_meta_owner_id ON document_metadata(owner_id);")
        self.conn.commit()

        self._lock = threading.Lock()
        self._migrate_if_needed()

    def _migrate_if_needed(self):
        # Fallback migration from JSON
        if os.path.exists(BM25_FILE_PATH) and not os.path.exists(SQLITE_DB_PATH):
            logger.info("Migrating JSON BM25 corpus to SQLite FTS5...")
            tmp_json = BM25_FILE_PATH + ".tmp_migration"
            try:
                os.rename(BM25_FILE_PATH, tmp_json)
                with open(tmp_json, "r") as f:
                    corpus = json.load(f)
                
                self.add_documents(corpus)
                os.remove(tmp_json)
                logger.info("Migration successful.")
            except Exception as e:
                logger.error(f"Failed to migrate BM25 corpus: {e}")
                if os.path.exists(tmp_json):
                    os.rename(tmp_json, BM25_FILE_PATH)

        # Ensure corpus_mapping is in sync with corpus (for upgrades to this schema)
        with self._lock:
            cursor = self.conn.execute("SELECT COUNT(*) FROM corpus_mapping")
            mapping_count = cursor.fetchone()[0]
            cursor = self.conn.execute("SELECT COUNT(*) FROM corpus")
            corpus_count = cursor.fetchone()[0]
            
            if mapping_count == 0 and corpus_count > 0:
                logger.info("Migrating FTS5 corpus to new corpus_mapping schema...")
                self.conn.execute("""
                    INSERT INTO corpus_mapping (rowid, chunk_id, doc_id)
                    SELECT rowid, id, doc_id FROM corpus
                """)
                self.conn.commit()
                logger.info("Mapping migration successful.")

    def add_documents(self, documents: List[Dict[str, str]]):
        if not documents:
            return
        with self._lock:
            for d in documents:
                chunk_id = d["id"]
                text = d["text"]
                doc_id = d.get("doc_id", "unknown")
                owner_id = str(d.get("owner_id", "") or "")
                
                # Delete existing chunk if present (using O(1) rowid lookup)
                cursor = self.conn.execute("SELECT rowid FROM corpus_mapping WHERE chunk_id = ?", (chunk_id,))
                row = cursor.fetchone()
                if row:
                    old_rowid = row[0]
                    self.conn.execute("DELETE FROM corpus WHERE rowid = ?", (old_rowid,))
                    self.conn.execute("DELETE FROM corpus_mapping WHERE rowid = ?", (old_rowid,))
                
                # Insert into FTS5
                cursor = self.conn.execute(
                    "INSERT INTO corpus (id, text, doc_id, owner_id) VALUES (?, ?, ?, ?)",
                    (chunk_id, text, doc_id, owner_id)
                )
                new_rowid = cursor.lastrowid
                
                # Insert into mapping
                self.conn.execute(
                    "INSERT INTO corpus_mapping (rowid, chunk_id, doc_id) VALUES (?, ?, ?)",
                    (new_rowid, chunk_id, doc_id)
                )
                
            self.conn.commit()
        logger.info(f"Added {len(documents)} documents to BM25 index.")

    def delete_documents_by_doc_id(self, doc_id: str, owner_id: Optional[str] = None):
        with self._lock:
            if owner_id is not None:
                # Find all rowids for this doc_id matching owner_id
                cursor = self.conn.execute(
                    "SELECT rowid FROM corpus WHERE doc_id = ? AND owner_id = ?",
                    (doc_id, str(owner_id))
                )
                rowids = [row[0] for row in cursor.fetchall()]
                for rid in rowids:
                    self.conn.execute("DELETE FROM corpus WHERE rowid = ?", (rid,))
                    self.conn.execute("DELETE FROM corpus_mapping WHERE rowid = ?", (rid,))
                self.conn.execute(
                    "DELETE FROM document_metadata WHERE doc_id = ? AND owner_id = ?",
                    (doc_id, str(owner_id))
                )
            else:
                # Find all rowids for this doc_id
                cursor = self.conn.execute("SELECT rowid FROM corpus_mapping WHERE doc_id = ?", (doc_id,))
                rowids = [row[0] for row in cursor.fetchall()]
                for rid in rowids:
                    self.conn.execute("DELETE FROM corpus WHERE rowid = ?", (rid,))
                self.conn.execute("DELETE FROM corpus_mapping WHERE doc_id = ?", (doc_id,))
                self.conn.execute("DELETE FROM document_metadata WHERE doc_id = ?", (doc_id,))
            self.conn.commit()
        logger.info(f"Deleted chunks from BM25 for doc_id: {doc_id}")

    def update_document_metadata(
        self,
        doc_id: str,
        filename: str,
        chunk_count: int,
        status: str,
        created_at: str,
        owner_id: Optional[str] = None
    ):
        with self._lock:
            self.conn.execute("""
                INSERT INTO document_metadata (doc_id, filename, chunk_count, status, created_at, owner_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename=excluded.filename,
                    chunk_count=excluded.chunk_count,
                    status=excluded.status,
                    created_at=excluded.created_at,
                    owner_id=COALESCE(excluded.owner_id, document_metadata.owner_id)
            """, (doc_id, filename, chunk_count, status, created_at, owner_id))
            self.conn.commit()

    def get_document_metadata(
        self,
        owner_id: Optional[str] = None,
        doc_id: Optional[str] = None
    ) -> List[Dict]:
        with self._lock:
            query = "SELECT doc_id, filename, chunk_count, status, created_at, owner_id FROM document_metadata"
            params = []
            conditions = []
            if owner_id is not None:
                conditions.append("owner_id = ?")
                params.append(owner_id)
            if doc_id is not None:
                conditions.append("doc_id = ?")
                params.append(doc_id)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            cursor = self.conn.execute(query, tuple(params))
            results = []
            for row in cursor:
                results.append({
                    "doc_id": row[0],
                    "filename": row[1],
                    "chunk_count": row[2],
                    "status": row[3],
                    "created_at": row[4],
                    "owner_id": row[5]
                })
            return results

    def get_document(self, doc_id: str) -> Optional[Dict]:
        docs = self.get_document_metadata(doc_id=doc_id)
        return docs[0] if docs else None

    def search(self, query: str, top_k: int = 5, owner_id: Optional[str] = None) -> List[Dict]:
        import re
        if not query.strip() or top_k <= 0:
            return []
            
        words = [w for w in re.split(r'\W+', query) if w]
        if not words:
            return []
        safe_query = " OR ".join(words)
        
        try:
            if owner_id is not None:
                cursor = self.conn.execute(
                    """
                    SELECT id, text, -bm25(corpus) as score 
                    FROM corpus 
                    WHERE corpus MATCH ? AND owner_id = ? 
                    ORDER BY score DESC 
                    LIMIT ?
                    """, 
                    (safe_query, str(owner_id), top_k)
                )
            else:
                cursor = self.conn.execute(
                    """
                    SELECT id, text, -bm25(corpus) as score 
                    FROM corpus 
                    WHERE corpus MATCH ? 
                    ORDER BY score DESC 
                    LIMIT ?
                    """, 
                    (safe_query, top_k)
                )
            results = []
            for row in cursor:
                if row[2] > 0:
                    results.append({"id": row[0], "text": row[1], "score": float(row[2])})
            return results
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS5 query failed (likely syntax error with input string): {e}")
            return []
            
    def get_total_docs(self) -> int:
        with self._lock:
            cursor = self.conn.execute("SELECT COUNT(*) FROM document_metadata")
            row = cursor.fetchone()
            return row[0] if row else 0

SQLiteBM25Store = BM25Store


class PostgresBM25Store:
    """PostgreSQL full-text-search sparse store using tsvector and GIN indexing.

    Supports multi-process concurrency (e.g. multi-worker Gunicorn) without
    file-locking constraints.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        use_pool: bool = False,
        minconn: int = 1,
        maxconn: int = 10,
    ):
        if psycopg2 is None:
            raise ImportError(
                "psycopg2 is required for PostgresBM25Store. "
                "Please install psycopg2 or psycopg2-binary."
            )
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL is not set. Please provide database_url or set the "
                "DATABASE_URL environment variable."
            )
        self.use_pool = use_pool
        self._minconn = minconn
        self._maxconn = maxconn
        self._pool = None
        self._pool_lock = threading.Lock()
        self._pid = os.getpid()

        if self.use_pool and ThreadedConnectionPool is not None:
            self._pool = ThreadedConnectionPool(self._minconn, self._maxconn, dsn=self.database_url)

        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Yield a database connection safe across processes and threads."""
        if self.use_pool and self._pool is not None:
            if os.getpid() != self._pid:
                with self._pool_lock:
                    if os.getpid() != self._pid:
                        try:
                            self._pool.closeall()
                        except Exception:
                            pass
                        self._pool = ThreadedConnectionPool(self._minconn, self._maxconn, dsn=self.database_url)
                        self._pid = os.getpid()
            conn = self._pool.getconn()
            try:
                yield conn
            except Exception:
                if conn and not conn.closed:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                raise
            finally:
                if conn and not conn.closed:
                    self._pool.putconn(conn)
        else:
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
        """Idempotently create tables and indexes."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Document metadata table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_metadata (
                        doc_id TEXT PRIMARY KEY,
                        filename TEXT,
                        chunk_count INTEGER,
                        status TEXT,
                        created_at TEXT,
                        owner_id TEXT
                    );
                """)

                # Guarded check: ensure owner_id column exists
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'document_metadata';
                """)
                doc_cols = {row[0] for row in cur.fetchall()}
                if doc_cols and "owner_id" not in doc_cols:
                    cur.execute("ALTER TABLE document_metadata ADD COLUMN owner_id TEXT;")

                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_doc_meta_owner_id ON document_metadata(owner_id);"
                )

                # 2. Corpus table with generated tsvector column
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS corpus (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT,
                        owner_id TEXT,
                        text TEXT,
                        ts tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED
                    );
                """)

                # Guarded check: ensure owner_id column exists if corpus table pre-existed
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'corpus';
                """)
                corpus_cols = {row[0] for row in cur.fetchall()}
                if corpus_cols and "owner_id" not in corpus_cols:
                    cur.execute("ALTER TABLE corpus ADD COLUMN owner_id TEXT;")

                # Guarded check: ensure ts column exists if corpus table pre-existed
                if corpus_cols and "ts" not in corpus_cols:
                    cur.execute("""
                        ALTER TABLE corpus ADD COLUMN ts tsvector
                        GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED;
                    """)

                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_corpus_ts ON corpus USING gin(ts);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_corpus_doc_id ON corpus(doc_id);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_corpus_owner_id ON corpus(owner_id);"
                )
            conn.commit()

    def add_documents(self, documents: List[Dict[str, str]]):
        if not documents:
            return
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                params = [
                    (
                        str(d["id"]),
                        str(d.get("doc_id", "unknown")),
                        str(d["owner_id"]) if d.get("owner_id") is not None else None,
                        d.get("text", "") or ""
                    )
                    for d in documents
                ]
                execute_batch(
                    cur,
                    """
                    INSERT INTO corpus (chunk_id, doc_id, owner_id, text)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        doc_id = EXCLUDED.doc_id,
                        owner_id = COALESCE(EXCLUDED.owner_id, corpus.owner_id),
                        text = EXCLUDED.text;
                    """,
                    params,
                )
            conn.commit()
        logger.info(f"Added {len(documents)} documents to BM25 index.")

    def delete_documents_by_doc_id(self, doc_id: str, owner_id: Optional[str] = None):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if owner_id is not None:
                    cur.execute(
                        "DELETE FROM corpus WHERE doc_id = %s AND owner_id = %s;",
                        (doc_id, str(owner_id))
                    )
                    cur.execute(
                        "DELETE FROM document_metadata WHERE doc_id = %s AND owner_id = %s;",
                        (doc_id, str(owner_id))
                    )
                else:
                    cur.execute("DELETE FROM corpus WHERE doc_id = %s;", (doc_id,))
                    cur.execute("DELETE FROM document_metadata WHERE doc_id = %s;", (doc_id,))
            conn.commit()
        logger.info(f"Deleted chunks from BM25 for doc_id: {doc_id}")

    def update_document_metadata(
        self,
        doc_id: str,
        filename: str,
        chunk_count: int,
        status: str,
        created_at: str,
        owner_id: Optional[str] = None,
    ):
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO document_metadata (doc_id, filename, chunk_count, status, created_at, owner_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        chunk_count = EXCLUDED.chunk_count,
                        status = EXCLUDED.status,
                        created_at = EXCLUDED.created_at,
                        owner_id = COALESCE(EXCLUDED.owner_id, document_metadata.owner_id);
                    """,
                    (doc_id, filename, chunk_count, status, created_at, owner_id),
                )
            conn.commit()

    def get_document_metadata(
        self,
        owner_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[Dict]:
        query = "SELECT doc_id, filename, chunk_count, status, created_at, owner_id FROM document_metadata"
        params = []
        conditions = []
        if owner_id is not None:
            conditions.append("owner_id = %s")
            params.append(owner_id)
        if doc_id is not None:
            conditions.append("doc_id = %s")
            params.append(doc_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "doc_id": row[0],
                        "filename": row[1],
                        "chunk_count": row[2],
                        "status": row[3],
                        "created_at": row[4],
                        "owner_id": row[5],
                    })
                return results

    def get_document(self, doc_id: str) -> Optional[Dict]:
        docs = self.get_document_metadata(doc_id=doc_id)
        return docs[0] if docs else None

    def search(self, query: str, top_k: int = 5, owner_id: Optional[str] = None) -> List[Dict]:
        if not query or not query.strip() or top_k <= 0:
            return []

        if owner_id is not None:
            sql = """
                SELECT chunk_id, text, ts_rank_cd(ts, query) AS score
                FROM corpus, websearch_to_tsquery('english', %s) query
                WHERE ts @@ query AND owner_id = %s
                ORDER BY score DESC
                LIMIT %s;
            """
            params = (query.strip(), str(owner_id), top_k)
        else:
            sql = """
                SELECT chunk_id, text, ts_rank_cd(ts, query) AS score
                FROM corpus, websearch_to_tsquery('english', %s) query
                WHERE ts @@ query
                ORDER BY score DESC
                LIMIT %s;
            """
            params = (query.strip(), top_k)

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    results = []
                    for row in rows:
                        score = float(row[2]) if row[2] is not None else 0.0
                        if score > 0:
                            results.append({
                                "id": row[0],
                                "text": row[1],
                                "score": score,
                            })
                    return results
        except Exception as e:
            logger.warning(f"Postgres text search query failed: {e}")
            return []

    def get_total_docs(self) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM document_metadata;")
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def close(self):
        """Close connection pool if enabled."""
        if self._pool is not None:
            try:
                self._pool.closeall()
            except Exception:
                pass


_bm25_store: Optional[Union[BM25Store, PostgresBM25Store]] = None
_current_backend: Optional[str] = None


def get_bm25_store() -> Union[BM25Store, PostgresBM25Store]:
    """Return the configured sparse store singleton.

    Controlled by SPARSE_BACKEND env var:
    - 'sqlite' (default): Returns SQLite FTS5 BM25Store.
    - 'postgres': Returns PostgreSQL PostgresBM25Store.
    """
    global _bm25_store, _current_backend
    backend = os.getenv("SPARSE_BACKEND", "sqlite").lower().strip()
    if _bm25_store is None or _current_backend != backend:
        if backend == "postgres":
            _bm25_store = PostgresBM25Store()
            _current_backend = "postgres"
        else:
            _bm25_store = BM25Store()
            _current_backend = "sqlite"
    return _bm25_store


def reset_bm25_store() -> None:
    """Reset cached singleton instance (useful for testing or backend switching)."""
    global _bm25_store, _current_backend
    _bm25_store = None
    _current_backend = None

