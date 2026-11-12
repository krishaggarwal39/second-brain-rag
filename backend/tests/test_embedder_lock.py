"""Unit tests for local embedder thread-safe load and cooldown retry (R-28)."""

import time
import pytest
from unittest.mock import patch, MagicMock

import app.rag.embeddings.local_embedder as embedder_module
from app.rag.embeddings.local_embedder import _get_model


class TestEmbedderLockAndCooldownR28:
    @pytest.fixture(autouse=True)
    def reset_embedder_state(self):
        embedder_module._model = None
        embedder_module._last_failed_time = 0.0
        yield
        embedder_module._model = None
        embedder_module._last_failed_time = 0.0

    def test_failed_load_enters_cooldown_and_retries_after_cooldown_expires(self):
        """R-28: Failed load enters 30s cooldown, skipping attempts until cooldown expires."""
        mock_sentence_transformer = MagicMock()

        with patch("sentence_transformers.SentenceTransformer", side_effect=[
            RuntimeError("HuggingFace download failed"),
            mock_sentence_transformer,
        ]) as mock_cls:
            # 1. First attempt fails
            model1 = _get_model()
            assert model1 is None
            assert mock_cls.call_count == 1
            assert embedder_module._last_failed_time > 0

            # 2. Immediate second attempt (within 30s cooldown) skips loading
            model2 = _get_model()
            assert model2 is None
            assert mock_cls.call_count == 1  # Not called again!

            # 3. Simulate cooldown expiration (31 seconds later)
            embedder_module._last_failed_time = time.time() - 31.0

            # 3. Third attempt retries and succeeds
            model3 = _get_model()
            assert model3 is mock_sentence_transformer
            assert mock_cls.call_count == 2
            assert embedder_module._last_failed_time == 0.0

    def test_thread_safe_concurrent_lazy_load(self):
        """R-28: Multiple concurrent threads calling _get_model load the model exactly once."""
        import concurrent.futures

        mock_sentence_transformer = MagicMock()

        def slow_init(*args, **kwargs):
            time.sleep(0.05)
            return mock_sentence_transformer

        with patch("sentence_transformers.SentenceTransformer", side_effect=slow_init) as mock_cls:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(_get_model) for _ in range(5)]
                results = [f.result() for f in futures]

            # All threads received the same model
            for res in results:
                assert res is mock_sentence_transformer

            # Loaded exactly once
            assert mock_cls.call_count == 1
