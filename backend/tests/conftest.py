"""Shared test fixtures for the Second Brain RAG backend."""

import os
import pytest

# Set test environment variables before any app imports
os.environ.setdefault("API_KEY", "test-api-key-12345")
os.environ.setdefault("API_KEYS", "test-api-key-12345,second-key-67890")
os.environ.setdefault("GOOGLE_API_KEY", "fake-gemini-key")
os.environ.setdefault("GROQ_API_KEY", "fake-groq-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("CACHE_BACKEND", "memory")
os.environ.setdefault("CHROMA_HOST", "localhost")
os.environ.setdefault("CHROMA_PORT", "8001")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-key-for-testing-only-12345")
os.environ.setdefault("JWT_EXPIRY_MINUTES", "1440")


@pytest.fixture
def auth_headers():
    """Valid API key and JWT bearer headers for authenticated endpoints."""
    from app.core.auth import create_access_token, get_user_store, pwd_context
    from datetime import datetime, timezone
    store = get_user_store()
    if not store.user_exists(1):
        with store._lock:
            cur = store.conn.execute("SELECT id FROM users WHERE id = 1")
            if not cur.fetchone():
                hashed = pwd_context.hash("strongpassword123")
                store.conn.execute(
                    "INSERT OR IGNORE INTO users (id, email, hashed_password, created_at) VALUES (1, ?, ?, ?)",
                    ("testuser1@example.com", hashed, datetime.now(timezone.utc).isoformat())
                )
                store.conn.commit()
    token = create_access_token(user_id=1)
    api_key = os.getenv("API_KEY", "test-api-key-12345")
    return {
        "X-API-Key": api_key,
        "Authorization": f"Bearer {token}",
    }


@pytest.fixture
def sample_documents():
    """Sample langchain Documents for testing chunking and ingestion."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables "
            "systems to learn from data. Deep learning uses neural networks with multiple layers.",
            metadata={"filename": "ml_intro.pdf", "page_number": 1, "source_type": "pdf"},
        ),
        Document(
            page_content="Python is a high-level programming language widely used in data science. "
            "Libraries like NumPy and Pandas provide powerful data manipulation tools.",
            metadata={"filename": "python_guide.pdf", "page_number": 1, "source_type": "pdf"},
        ),
    ]


@pytest.fixture
def mock_embeddings():
    """Mock embedding vectors (384-dim like all-MiniLM-L6-v2)."""
    import numpy as np

    return [np.random.rand(384).tolist() for _ in range(5)]


@pytest.fixture
def mock_chroma_results():
    """Mock ChromaDB query results."""
    return [
        {
            "text": "Machine learning is a subset of AI.",
            "metadata": {
                "filename": "ml_intro.pdf",
                "page_number": 1,
                "hash": "abc123",
                "parent_id": "parent-uuid-1",
            },
            "score": 0.85,
        },
        {
            "text": "Deep learning uses neural networks.",
            "metadata": {
                "filename": "ml_intro.pdf",
                "page_number": 2,
                "hash": "def456",
                "parent_id": "parent-uuid-2",
            },
            "score": 0.72,
        },
    ]


@pytest.fixture
def mock_bm25_results():
    """Mock BM25 search results."""
    return [
        {"id": "bm25-1", "text": "Neural networks are used in deep learning.", "score": 3.5},
        {"id": "bm25-2", "text": "Machine learning algorithms learn from data.", "score": 2.8},
    ]
