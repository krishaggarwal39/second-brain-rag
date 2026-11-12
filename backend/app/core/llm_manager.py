import os
import threading
import time
import asyncio
from typing import Optional, Dict, Any, List

import httpx

from app.core.logging import get_logger, sanitize_error_msg

logger = get_logger(__name__)


class LLMManager:
    """
    Manages LLM providers (Gemini primary, Groq fallback) with automatic
    failover on 429 rate-limit errors and cooldown-based recovery.

    Provides a unified `generate()` method that eliminates the need for
    callers to build provider-specific payloads or handle retries.
    """

    def __init__(self):
        self._lock = threading.Lock()

        self.primary_provider = "gemini"
        self.fallback_provider = "groq"
        self.active_provider = self.primary_provider

        self.last_fallback_time = 0
        self.cooldown_seconds = int(os.getenv("LLM_COOLDOWN_SECONDS", "300"))

        # GOOGLE_API_KEY may be a comma-separated list of keys (different accounts)
        # for quota rotation. On a 429 we advance to the next key before failing over
        # to the Groq provider. Backward compatible with a single key.
        self.gemini_keys = [k.strip() for k in os.getenv("GOOGLE_API_KEY", "").split(",") if k.strip()]
        self._gemini_key_idx = 0
        self._key_cooldown_until: Dict[str, float] = {}
        self.groq_key = os.getenv("GROQ_API_KEY", "")

        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        if not self.gemini_keys:
            logger.warning("GOOGLE_API_KEY is not set.")
        else:
            logger.info(f"Loaded {len(self.gemini_keys)} Gemini API key(s) for rotation.")
        if not self.groq_key:
            logger.warning("GROQ_API_KEY is not set.")

    @property
    def gemini_key(self) -> str:
        """Currently active Gemini key."""
        return self.gemini_keys[self._gemini_key_idx] if self.gemini_keys else ""

    def rotate_gemini_key(self, failed_key: Optional[str] = None) -> bool:
        """Handle a 429 for `failed_key` (default: the current key). R-28.

        Returns True when the caller should retry with the (possibly new) current
        key, False when every key is cooling down (caller fails over to Groq).

        - Concurrent 429s for the same key rotate only once: if another request
          already moved past `failed_key`, we just retry with the current key.
        - A rate-limited key cools down for GEMINI_KEY_COOLDOWN_SECONDS and is
          then eligible again, so rotation never gets stuck at the last key.
        """
        cooldown = float(os.getenv("GEMINI_KEY_COOLDOWN_SECONDS", "60"))
        with self._lock:
            if not self.gemini_keys:
                return False
            now = time.time()
            if failed_key is None:
                failed_key = self.gemini_keys[self._gemini_key_idx]
            self._key_cooldown_until[failed_key] = now + cooldown
            current = self.gemini_keys[self._gemini_key_idx]
            if current != failed_key and self._key_cooldown_until.get(current, 0) <= now:
                return True  # someone else already rotated to a healthy key
            n = len(self.gemini_keys)
            for step in range(1, n + 1):
                idx = (self._gemini_key_idx + step) % n
                key = self.gemini_keys[idx]
                if self._key_cooldown_until.get(key, 0) <= now:
                    if idx != self._gemini_key_idx:
                        self._gemini_key_idx = idx
                        logger.warning(f"Gemini key rate-limited; rotating to key {idx + 1}/{n}")
                        return True
            logger.warning("All Gemini keys are cooling down; no more keys to rotate to.")
            return False

    @staticmethod
    def generation_settings() -> Dict[str, Any]:
        """Sampling settings shared by every answer call (R-26)."""
        return {
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
            "max_output_tokens": int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024")),
        }

    def get_active_provider(self) -> str:
        with self._lock:
            if self.active_provider != self.primary_provider:
                if time.time() - self.last_fallback_time > self.cooldown_seconds:
                    logger.info(f"Cooldown expired. Reverting to primary LLM: {self.primary_provider}")
                    self.active_provider = self.primary_provider
            return self.active_provider

    def get_api_key(self, provider: str) -> str:
        if provider == "gemini":
            return self.gemini_key
        elif provider == "groq":
            return self.groq_key
        return ""

    def switch_to_fallback(self, current_failing_provider: str):
        with self._lock:
            if self.active_provider == current_failing_provider:
                if self.active_provider == self.primary_provider and self.groq_key:
                    logger.warning(
                        f"Rate limit hit! Switching from {self.primary_provider} to {self.fallback_provider}"
                    )
                    self.active_provider = self.fallback_provider
                    self.last_fallback_time = time.time()
                elif self.active_provider == self.fallback_provider and self.gemini_key:
                    logger.warning(
                        f"Fallback rate limit hit! Switching back to {self.primary_provider}"
                    )
                    self.active_provider = self.primary_provider
                    self.last_fallback_time = time.time()
                else:
                    logger.error(
                        f"Cannot fallback — active: {self.active_provider}, backup key missing."
                    )

    # ------------------------------------------------------------------
    # Unified generation interface
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        provider: str,
        system_prompt: str,
        user_content: str,
        json_mode: bool = False,
    ) -> tuple[str, Dict[str, str], Dict[str, Any]]:
        """Build provider-specific URL, headers, and payload."""
        api_key = self.get_api_key(provider)

        if provider == "gemini":
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.gemini_model}:generateContent?key={api_key}"
            )
            headers = {"Content-Type": "application/json"}
            gen_config: Dict[str, Any] = {}
            if json_mode:
                gen_config["responseMimeType"] = "application/json"

            payload: Dict[str, Any] = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_content}]}],
            }
            if gen_config:
                payload["generationConfig"] = gen_config

            return url, headers, payload

        else:  # groq
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            payload = {
                "model": self.groq_model,
                "messages": messages,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            return url, headers, payload

    @staticmethod
    def _extract_text(provider: str, response_data: Dict[str, Any]) -> Optional[str]:
        """Extract the generated text from a provider's response JSON."""
        if provider == "gemini":
            candidates = response_data.get("candidates", [])
            if candidates:
                return candidates[0]["content"]["parts"][0]["text"]
            return None
        else:
            choices = response_data.get("choices", [])
            if choices:
                return choices[0]["message"]["content"]
            return None

    async def generate(
        self,
        system_prompt: str,
        user_content: str,
        json_mode: bool = False,
        timeout: float = 30.0,
    ) -> Optional[str]:
        """
        Unified LLM generation with automatic retry and provider fallback.

        Args:
            system_prompt: The system instruction for the LLM.
            user_content: The user message content.
            json_mode: If True, requests structured JSON output.
            timeout: HTTP request timeout in seconds.

        Returns:
            The generated text string, or None on failure.
        """
        max_retries = int(os.getenv("MAX_API_RETRIES", "3"))

        for attempt in range(max_retries):
            provider = self.get_active_provider()
            api_key = self.get_api_key(provider)

            if not api_key:
                logger.error(f"{provider} API key not set — cannot generate.")
                return None

            url, headers, payload = self._build_payload(
                provider, system_prompt, user_content, json_mode
            )

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        url, headers=headers, json=payload, timeout=timeout
                    )

                    if response.status_code == 429:
                        # On Gemini 429, first try the next Gemini key (rotation);
                        # only fail over to Groq once all Gemini keys are exhausted.
                        if provider == "gemini" and self.rotate_gemini_key(api_key):
                            continue
                        self.switch_to_fallback(provider)
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue

                    response.raise_for_status()
                    data = response.json()
                    result = self._extract_text(provider, data)

                    if result is None:
                        logger.warning(f"No content in {provider} response.")
                    return result

            except httpx.HTTPStatusError as e:
                if e.response.status_code != 429:
                    logger.error(
                        f"LLM HTTP error ({provider}): "
                        f"{sanitize_error_msg(str(e))}"
                    )
                    break
            except Exception as e:
                logger.error(f"LLM generation failed ({provider}): {e}")
                break

        return None


llm_manager = LLMManager()
