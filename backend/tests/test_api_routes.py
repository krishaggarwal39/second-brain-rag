"""Tests for API routes (health, documents, chat)."""

import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    # Patch heavy dependencies before importing the app
    with patch("app.rag.retrievers.hybrid_retriever.CrossEncoder"), \
         patch("app.core.cache.init_cache", new_callable=AsyncMock), \
         patch("app.core.cache.close_cache", new_callable=AsyncMock), \
         patch("app.rag.vectorstore.chroma_store.get_stats", return_value={"total_chunks": 10}):
        from app.main import app
        with TestClient(app) as c:
            yield c


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


class TestHealthRoutes:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Second Brain RAG" in response.json()["message"]

    def test_cors_preflight_allows_authorization(self, client):
        response = client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization, content-type",
            },
        )
        assert response.status_code == 200
        allow_headers = response.headers.get("access-control-allow-headers", "").lower()
        assert "authorization" in allow_headers


class TestAuthMiddleware:
    def test_protected_route_without_key(self, client):
        response = client.get("/documents")
        assert response.status_code == 403

    def test_protected_route_with_invalid_key(self, client):
        response = client.get("/documents", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 403

    def test_protected_route_with_valid_key(self, client, auth_headers):
        with patch(
            "app.api.routes.documents.get_bm25_store"
        ) as mock_bm25:
            mock_store = MagicMock()
            mock_store.get_document_metadata.return_value = []
            mock_bm25.return_value = mock_store

            response = client.get("/documents", headers=auth_headers)
            assert response.status_code == 200

    def test_second_api_key_is_accepted(self, client):
        """Multi-key auth: the second key in API_KEYS should also work."""
        from app.core.auth import create_access_token
        token = create_access_token(user_id=1)
        with patch(
            "app.api.routes.documents.get_bm25_store"
        ) as mock_bm25, patch(
            "app.main._valid_keys", {os.getenv("API_KEY", "test-api-key-12345"), "second-key-67890"}
        ):
            mock_store = MagicMock()
            mock_store.get_document_metadata.return_value = []
            mock_bm25.return_value = mock_store

            response = client.get(
                "/documents",
                headers={
                    "X-API-Key": "second-key-67890",
                    "Authorization": f"Bearer {token}",
                }
            )
            assert response.status_code == 200


class TestDocumentRoutes:
    def test_upload_invalid_extension(self, client, auth_headers):
        """Only PDF and DOCX should be accepted."""
        response = client.post(
            "/documents/upload",
            headers=auth_headers,
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400

    def test_get_job_status_unknown(self, client, auth_headers):
        with patch("app.api.routes.documents.get_cache", new_callable=AsyncMock) as mock_cache:
            mock_cache.return_value = None
            response = client.get("/documents/jobs/fake-uuid", headers=auth_headers)
            assert response.status_code == 200
            assert response.json()["status"] == "unknown"

    def test_get_job_status_completed(self, client, auth_headers):
        with patch("app.api.routes.documents.get_cache", new_callable=AsyncMock) as mock_cache:
            mock_cache.return_value = "completed"
            response = client.get("/documents/jobs/real-uuid", headers=auth_headers)
            assert response.status_code == 200
            assert response.json()["status"] == "completed"

    def test_list_documents_empty(self, client, auth_headers):
        with patch(
            "app.api.routes.documents.get_bm25_store"
        ) as mock_bm25:
            mock_store = MagicMock()
            mock_store.get_document_metadata.return_value = []
            mock_bm25.return_value = mock_store

            response = client.get("/documents", headers=auth_headers)
            assert response.status_code == 200
            assert response.json()["documents"] == []

    def test_documents_requires_jwt(self, client):
        """API key alone is not enough; JWT bearer token is also required."""
        api_key = os.getenv("API_KEY", "test-api-key-12345")
        response = client.get("/documents", headers={"X-API-Key": api_key})
        assert response.status_code == 401

    def test_list_documents_filters_by_owner(self, client, auth_headers):
        with patch("app.api.routes.documents.get_bm25_store") as mock_bm25:
            mock_store = MagicMock()
            mock_store.get_document_metadata.return_value = [
                {"doc_id": "doc-1", "filename": "doc-1.pdf", "chunk_count": 1, "status": "completed", "created_at": "now", "owner_id": "1"}
            ]
            mock_bm25.return_value = mock_store

            response = client.get("/documents", headers=auth_headers)
            assert response.status_code == 200
            assert len(response.json()["documents"]) == 1
            mock_store.get_document_metadata.assert_called_with("1")

    def test_delete_doc_owner_success(self, client, auth_headers):
        with patch("app.api.routes.documents.get_bm25_store") as mock_bm25, \
             patch("app.api.routes.documents.get_document_parents", new_callable=AsyncMock, return_value=[]) as mock_parents, \
             patch("app.api.routes.documents.delete_document", new_callable=AsyncMock) as mock_delete:
            mock_store = MagicMock()
            mock_store.get_document_metadata.return_value = [
                {"doc_id": "doc-1", "owner_id": "1"}
            ]
            mock_bm25.return_value = mock_store

            response = client.delete("/documents/doc-1", headers=auth_headers)
            assert response.status_code == 200
            assert response.json()["status"] == "deleted"
            mock_parents.assert_called_once_with("doc-1", owner_id="1")
            mock_delete.assert_called_once_with("doc-1", owner_id="1")
            mock_store.delete_documents_by_doc_id.assert_called_once_with("doc-1", owner_id="1")

    def test_upload_url_generates_scoped_doc_id(self, client, auth_headers):
        with patch("app.worker.process_url_task.delay") as mock_delay:
            response = client.post(
                "/documents/url",
                headers=auth_headers,
                json={"url": "https://example.com/test"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "processing"
            assert "job_id" in data
            assert "doc_id" in data
            assert data["doc_id"].startswith("1:")
            mock_delay.assert_called_once_with(
                "https://example.com/test", data["job_id"], "1", data["doc_id"]
            )

    def test_delete_doc_other_owner_forbidden(self, client, auth_headers):
        with patch("app.api.routes.documents.get_bm25_store") as mock_bm25:
            mock_store = MagicMock()
            mock_store.get_document_metadata.return_value = [
                {"doc_id": "doc-2", "owner_id": "2"}
            ]
            mock_bm25.return_value = mock_store

            response = client.delete("/documents/doc-2", headers=auth_headers)
            assert response.status_code == 403

    def test_delete_doc_not_found_forbidden(self, client, auth_headers):
        with patch("app.api.routes.documents.get_bm25_store") as mock_bm25:
            mock_store = MagicMock()
            mock_store.get_document_metadata.return_value = []
            mock_bm25.return_value = mock_store

            response = client.delete("/documents/missing-doc", headers=auth_headers)
            assert response.status_code == 403

    def test_cache_generation_bumped_on_document_changes(self, client, auth_headers):
        with patch("app.api.routes.documents.bump_cache_generation", new_callable=AsyncMock) as mock_bump, \
             patch("app.api.routes.documents.get_bm25_store") as mock_bm25, \
             patch("app.api.routes.documents.get_document_parents", new_callable=AsyncMock, return_value=[]), \
             patch("app.api.routes.documents.delete_document", new_callable=AsyncMock), \
             patch("app.worker.process_url_task.delay"), \
             patch("app.worker.process_file_task.delay"), \
             patch("magic.from_file", return_value="application/pdf"):

            # 1. URL upload bumps generation for owner "1"
            res = client.post("/documents/url", headers=auth_headers, json={"url": "https://example.com/test"})
            assert res.status_code == 200
            mock_bump.assert_called_with("1")

            mock_bump.reset_mock()

            # 2. File upload bumps generation for owner "1"
            res = client.post(
                "/documents/upload",
                headers=auth_headers,
                files={"file": ("doc.pdf", b"%PDF-1.4 dummy", "application/pdf")},
            )
            assert res.status_code == 200
            mock_bump.assert_called_with("1")

            mock_bump.reset_mock()

            # 3. Delete document bumps generation for owner "1"
            mock_store = MagicMock()
            mock_store.get_document_metadata.return_value = [{"doc_id": "doc-1", "owner_id": "1"}]
            mock_bm25.return_value = mock_store

            res = client.delete("/documents/doc-1", headers=auth_headers)
            assert res.status_code == 200
            mock_bump.assert_called_with("1")



class TestChatRoutes:
    def test_chat_requires_auth(self, client):
        response = client.post("/chat", json={"question": "What is AI?"})
        assert response.status_code == 403

    def test_chat_validates_question_length(self, client, auth_headers):
        response = client.post(
            "/chat",
            headers=auth_headers,
            json={"question": ""},
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_chat_stream_requires_auth(self, client):
        response = client.post("/chat/stream", json={"question": "Hello"})
        assert response.status_code == 403

    def test_chat_validates_history_length(self, client, auth_headers):
        """Chat history should respect max_length=20."""
        long_history = [{"role": "user", "content": f"msg {i}"} for i in range(25)]

        response = client.post(
            "/chat",
            headers=auth_headers,
            json={"question": "test", "chat_history": long_history},
        )
        assert response.status_code == 422

    def test_chat_requires_jwt(self, client):
        """API key alone is not enough for chat; JWT bearer token is also required."""
        api_key = os.getenv("API_KEY", "test-api-key-12345")
        response = client.post(
            "/chat",
            headers={"X-API-Key": api_key},
            json={"question": "What is AI?"},
        )
        assert response.status_code == 401

    def test_chat_stream_requires_jwt(self, client):
        """API key alone is not enough for chat stream; JWT bearer token is also required."""
        api_key = os.getenv("API_KEY", "test-api-key-12345")
        response = client.post(
            "/chat/stream",
            headers={"X-API-Key": api_key},
            json={"question": "Hello"},
        )
        assert response.status_code == 401

    def test_chat_passes_owner_id(self, client, auth_headers):
        with patch("app.api.routes.chat.pipeline") as mock_pipeline:
            mock_pipeline.ask = AsyncMock(return_value={"answer": "42", "citations": []})
            response = client.post(
                "/chat",
                headers=auth_headers,
                json={"question": "What is life?"},
            )
            assert response.status_code == 200
            assert response.json()["answer"] == "42"
            mock_pipeline.ask.assert_called_once_with(
                "What is life?", [], owner_id="1"
            )


class TestAuthRoutes:
    def test_register_and_login_flow(self, client):
        import uuid
        test_email = f"user_{uuid.uuid4()}@example.com"
        # Register user
        reg_resp = client.post("/auth/register", json={
            "email": test_email,
            "password": "strongpassword123"
        })
        assert reg_resp.status_code == 200
        assert "access_token" in reg_resp.json()

        # Duplicate register should fail with 409
        dup_resp = client.post("/auth/register", json={
            "email": test_email,
            "password": "strongpassword123"
        })
        assert dup_resp.status_code == 409

        # Login with valid credentials
        login_resp = client.post("/auth/login", json={
            "email": test_email,
            "password": "strongpassword123"
        })
        assert login_resp.status_code == 200
        assert "access_token" in login_resp.json()

        # Login with wrong password
        bad_login = client.post("/auth/login", json={
            "email": test_email,
            "password": "wrongpassword"
        })
        assert bad_login.status_code == 401

    def test_login_unknown_email_returns_401(self, client):
        resp = client.post("/auth/login", json={
            "email": "nonexistent_unknown_user_99999@example.com",
            "password": "wrongpassword123"
        })
        assert resp.status_code == 401

    def test_token_for_nonexistent_user_returns_401(self, client):
        from app.core.auth import create_access_token
        fake_token = create_access_token(user_id=99999999)
        api_key = os.getenv("API_KEY", "test-api-key-12345")
        resp = client.get("/documents", headers={
            "X-API-Key": api_key,
            "Authorization": f"Bearer {fake_token}",
        })
        assert resp.status_code == 401

