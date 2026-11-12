"""Unit tests for upload hand-off, staging directory, and cleanup (R-02)."""

import os
import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.core.upload import get_upload_dir, cleanup_old_uploads
from app.main import app


class TestUploadDirAndCleanup:
    def test_get_upload_dir_creates_directory_and_permissions(self, tmp_path, monkeypatch):
        custom_dir = str(tmp_path / "custom_uploads")
        monkeypatch.setenv("UPLOAD_DIR", custom_dir)

        resolved = get_upload_dir()
        assert resolved == custom_dir
        assert os.path.isdir(custom_dir)
        # Check permissions: 0700 (owner rwx only)
        mode = oct(os.stat(custom_dir).st_mode & 0o777)
        assert mode == oct(0o700)

    def test_cleanup_old_uploads_deletes_files_older_than_24h_only(self, tmp_path):
        upload_dir = tmp_path / "cleanup_test"
        upload_dir.mkdir(mode=0o700)

        # Create old file (older than 24h = 86400s)
        old_file = upload_dir / "old_upload.pdf"
        old_file.write_bytes(b"%PDF-1.4 old")
        two_days_ago = time.time() - (2 * 86400)
        os.utime(str(old_file), (two_days_ago, two_days_ago))

        # Create recent file (< 24h)
        recent_file = upload_dir / "recent_upload.pdf"
        recent_file.write_bytes(b"%PDF-1.4 recent")

        deleted = cleanup_old_uploads(str(upload_dir), max_age_seconds=86400)
        assert deleted == 1
        assert not old_file.exists()
        assert recent_file.exists()


class TestUploadRoutesR02:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        from app.core.auth import create_access_token
        token = create_access_token(user_id=1, role="user")
        return {"Authorization": f"Bearer {token}", "X-API-Key": "test-api-key-12345"}

    def test_upload_uses_random_uuid_filename_not_client_filename(self, client, auth_headers, tmp_path, monkeypatch):
        upload_dir = str(tmp_path / "staging")
        monkeypatch.setenv("UPLOAD_DIR", upload_dir)

        enqueued_paths = []

        def fake_delay(path, safe_name, *args, **kwargs):
            enqueued_paths.append((path, safe_name))
            return MagicMock(id="task-123")

        with patch("app.worker.process_file_task.delay", side_effect=fake_delay), \
             patch("magic.from_file", return_value="application/pdf"):
            res = client.post(
                "/documents/upload",
                headers=auth_headers,
                files={"file": ("My Secret Document (Draft).pdf", b"%PDF-1.4 content", "application/pdf")},
            )
            assert res.status_code == 200
            assert len(enqueued_paths) == 1
            staged_path, original_name = enqueued_paths[0]
            assert original_name == "My Secret Document (Draft).pdf"

            # Path must NOT contain client's filename, must end with .pdf in upload_dir
            assert "My Secret Document" not in staged_path
            assert staged_path.startswith(upload_dir)
            assert staged_path.endswith(".pdf")
            basename = os.path.basename(staged_path)
            # Must be a uuid + .pdf (36 uuid chars + 4 = 40)
            assert len(basename) == 40

    def test_upload_deletes_file_and_returns_503_if_delay_raises(self, client, auth_headers, tmp_path, monkeypatch):
        upload_dir = str(tmp_path / "staging")
        monkeypatch.setenv("UPLOAD_DIR", upload_dir)

        with patch("app.worker.process_file_task.delay", side_effect=RuntimeError("Celery broker connection failed")), \
             patch("magic.from_file", return_value="application/pdf"):
            res = client.post(
                "/documents/upload",
                headers=auth_headers,
                files={"file": ("test.pdf", b"%PDF-1.4 content", "application/pdf")},
            )
            assert res.status_code == 503
            assert "Task queue unavailable" in res.json()["detail"]

            # Staged file must not be leaked in UPLOAD_DIR
            files_in_dir = os.listdir(upload_dir)
            assert len(files_in_dir) == 0
