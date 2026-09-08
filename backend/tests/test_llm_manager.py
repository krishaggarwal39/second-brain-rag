"""Tests for the LLM Manager (provider fallback and unified generation)."""

import time
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.core.llm_manager import LLMManager


class TestProviderFallback:
    def setup_method(self):
        with patch.dict(
            "os.environ",
            {"GOOGLE_API_KEY": "test-gemini", "GROQ_API_KEY": "test-groq"},
        ):
            self.manager = LLMManager()

    def test_default_provider_is_gemini(self):
        assert self.manager.get_active_provider() == "gemini"

    def test_switch_to_fallback_on_primary_failure(self):
        self.manager.switch_to_fallback("gemini")
        assert self.manager.get_active_provider() == "groq"

    def test_switch_back_on_fallback_failure(self):
        self.manager.switch_to_fallback("gemini")
        assert self.manager.get_active_provider() == "groq"

        self.manager.switch_to_fallback("groq")
        assert self.manager.get_active_provider() == "gemini"

    def test_no_switch_if_different_provider_fails(self):
        """Only switch if the currently active provider is the one that failed."""
        self.manager.switch_to_fallback("groq")  # groq is not active
        assert self.manager.get_active_provider() == "gemini"

    def test_cooldown_reverts_to_primary(self):
        self.manager.switch_to_fallback("gemini")
        assert self.manager.get_active_provider() == "groq"

        # Simulate cooldown expiry
        self.manager.last_fallback_time = time.time() - 400
        assert self.manager.get_active_provider() == "gemini"

    def test_get_api_key(self):
        assert self.manager.get_api_key("gemini") == "test-gemini"
        assert self.manager.get_api_key("groq") == "test-groq"
        assert self.manager.get_api_key("unknown") == ""


class TestPayloadBuilding:
    def setup_method(self):
        with patch.dict(
            "os.environ",
            {"GOOGLE_API_KEY": "test-key", "GROQ_API_KEY": "test-groq"},
        ):
            self.manager = LLMManager()

    def test_gemini_payload_structure(self):
        url, headers, payload = self.manager._build_payload(
            "gemini", "You are helpful.", "Hello", json_mode=True
        )

        assert "generativelanguage.googleapis.com" in url
        assert "key=test-key" in url
        assert headers["Content-Type"] == "application/json"
        assert "systemInstruction" in payload
        assert "contents" in payload
        assert payload["generationConfig"]["responseMimeType"] == "application/json"

    def test_groq_payload_structure(self):
        url, headers, payload = self.manager._build_payload(
            "groq", "You are helpful.", "Hello", json_mode=True
        )

        assert "api.groq.com" in url
        assert "Bearer test-groq" in headers["Authorization"]
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["response_format"]["type"] == "json_object"

    def test_gemini_no_json_mode(self):
        _, _, payload = self.manager._build_payload(
            "gemini", "System", "User", json_mode=False
        )
        assert "generationConfig" not in payload

    def test_groq_no_json_mode(self):
        _, _, payload = self.manager._build_payload(
            "groq", "System", "User", json_mode=False
        )
        assert "response_format" not in payload


class TestResponseExtraction:
    def test_extract_gemini_response(self):
        data = {
            "candidates": [
                {"content": {"parts": [{"text": "Hello world"}]}}
            ]
        }
        result = LLMManager._extract_text("gemini", data)
        assert result == "Hello world"

    def test_extract_groq_response(self):
        data = {
            "choices": [{"message": {"content": "Hello from Groq"}}]
        }
        result = LLMManager._extract_text("groq", data)
        assert result == "Hello from Groq"

    def test_extract_empty_gemini(self):
        result = LLMManager._extract_text("gemini", {"candidates": []})
        assert result is None

    def test_extract_empty_groq(self):
        result = LLMManager._extract_text("groq", {"choices": []})
        assert result is None


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_success(self):
        with patch.dict(
            "os.environ",
            {"GOOGLE_API_KEY": "key", "GROQ_API_KEY": "key2"},
        ):
            manager = LLMManager()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"entity": "Python"}'}]}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await manager.generate(
                system_prompt="Extract entity",
                user_content="What is Python?",
                json_mode=True,
            )

        assert result == '{"entity": "Python"}'

    @pytest.mark.asyncio
    async def test_generate_returns_none_without_key(self):
        with patch.dict(
            "os.environ",
            {"GOOGLE_API_KEY": "", "GROQ_API_KEY": ""},
        ):
            manager = LLMManager()

        result = await manager.generate("system", "user")
        assert result is None
