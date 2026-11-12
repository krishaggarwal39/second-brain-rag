import os
import time
import tempfile
import structlog
from typing import Optional

logger = structlog.get_logger(__name__)


def get_upload_dir() -> str:
    """Return the configured upload directory, ensuring it exists with mode 0700.

    R-02: Writes uploads to a shared configurable directory rather than
    container-local /tmp.
    """
    default_dir = os.path.join(tempfile.gettempdir(), "second_brain_uploads")
    upload_dir = os.getenv("UPLOAD_DIR", default_dir)
    os.makedirs(upload_dir, exist_ok=True)
    try:
        os.chmod(upload_dir, 0o700)
    except Exception as e:
        logger.debug("Failed to set 0700 permissions on upload dir", error=str(e))
    return upload_dir


def cleanup_old_uploads(upload_dir: Optional[str] = None, max_age_seconds: int = 86400) -> int:
    """Delete files in UPLOAD_DIR older than max_age_seconds (default 24h).

    Safe, lightweight cleanup helper called on API startup and worker task start (R-02).
    """
    if upload_dir is None:
        try:
            upload_dir = get_upload_dir()
        except Exception as e:
            logger.warning("Could not resolve upload dir for cleanup", error=str(e))
            return 0

    if not os.path.isdir(upload_dir):
        return 0

    now = time.time()
    deleted_count = 0
    try:
        for entry in os.scandir(upload_dir):
            if entry.is_file():
                try:
                    stat = entry.stat()
                    if (now - stat.st_mtime) > max_age_seconds:
                        os.remove(entry.path)
                        deleted_count += 1
                except OSError as e:
                    logger.debug("Failed to remove old upload file", path=entry.path, error=str(e))
    except Exception as e:
        logger.warning("Error during upload cleanup", error=str(e))

    if deleted_count > 0:
        logger.info("Cleaned up old upload files", count=deleted_count)
    return deleted_count
