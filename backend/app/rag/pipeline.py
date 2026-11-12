import os
import time
import json
import httpx
import asyncio
import structlog
import hashlib
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple

from app.rag.retrievers.hybrid_retriever import hybrid_search
from app.rag.graph.graph_retriever import retrieve_graph_context
from app.rag.context_engineering import (
    REFUSAL_TEXT,
    build_prompt_with_sources,
    neutralize_markup,
    prune_irrelevant_context,
)
from app.core.cache import daily_key, increment_metric, get_cache, set_cache, get_cache_generation
from app.core.llm_manager import llm_manager
from app.core.logging import sanitize_error_msg
from app.core.telemetry import get_tracer

logger = structlog.get_logger(__name__)

KNOWN_ERROR_PLACEHOLDERS = {
    "No response from Gemini API.",
    "Error generating answer. Check server logs.",
    "Streaming error occurred. Check server logs.",
    "\n\n⚠️ **API Rate Limit Exceeded.** You've hit the rate limits for all fallback providers.",
}

_REWRITE_SYSTEM = (
    "Rewrite the user's latest question into a single standalone search query, "
    "resolving pronouns and references using the conversation. Output only the query, "
    "no explanation. If it is already standalone, output it unchanged."
)


def is_error_placeholder(text: str) -> bool:
    """Check if the response text matches a known error placeholder."""
    if not text or not text.strip():
        return True
    if text in KNOWN_ERROR_PLACEHOLDERS:
        return True
    if text.startswith("API Error (") and "Check server logs." in text:
        return True
    if text.startswith("Stream error (") and "Check server logs." in text:
        return True
    return False


def _max_retries() -> int:
    # R-48: MAX_API_RETRIES=0 used to skip the loop and crash on an undefined payload.
    try:
        return max(1, int(os.getenv("MAX_API_RETRIES", "3")))
    except ValueError:
        return 3


def _price_per_million(provider: str, kind: str) -> float:
    """Configurable price per 1M tokens (R-27).

    COST_PER_1M_<INPUT|OUTPUT>_<PROVIDER> wins; otherwise the legacy flat
    COST_PER_1M_TOKENS (default 1.25) applies to every token.
    """
    specific = os.getenv(f"COST_PER_1M_{kind}_{provider.upper()}")
    if specific not in (None, ""):
        try:
            return float(specific)
        except ValueError:
            pass
    try:
        return float(os.getenv("COST_PER_1M_TOKENS", "1.25"))
    except ValueError:
        return 1.25


def _usage_from(provider: str, data: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """(input_tokens, output_tokens) reported by the provider, if any (R-27)."""
    if not isinstance(data, dict):
        return None
    if provider == "gemini":
        u = data.get("usageMetadata") or {}
        if u:
            return int(u.get("promptTokenCount", 0) or 0), int(u.get("candidatesTokenCount", 0) or 0)
        return None
    u = data.get("usage") or (data.get("x_groq") or {}).get("usage") or {}
    if u:
        return int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0)
    return None


class RAGPipeline:
    """Retrieval + generation. Configuration is read from the environment per call."""

    # ------------------------------------------------------------------ retrieval
    async def _rewrite_query(self, question: str, chat_history: List[Dict]) -> str:
        """Turn a follow-up into a standalone query for retrieval (R-29).

        The original question is still used for the answer prompt. Falls back
        to the original question on any error or timeout.
        """
        if not chat_history or os.getenv("QUERY_REWRITE", "true").lower() not in ("1", "true", "yes"):
            return question
        convo = "\n".join(
            f"{m.get('role', 'user')}: {str(m.get('content', ''))[:500]}" for m in chat_history[-6:]
        )
        user_content = f"Conversation:\n{convo}\n\nLatest question: {question}"
        try:
            rewritten = await asyncio.wait_for(
                llm_manager.generate(_REWRITE_SYSTEM, user_content, timeout=3.0),
                timeout=float(os.getenv("QUERY_REWRITE_TIMEOUT", "3")),
            )
        except Exception as e:
            logger.info("Query rewrite skipped", reason=type(e).__name__)
            return question
        rewritten = " ".join((rewritten or "").split())[:300]
        return rewritten or question

    async def _retrieve_all_context(
        self, question: str, owner_id: Optional[str] = None
    ) -> tuple[List[Dict], str, float]:
        tracer = get_tracer("rag.pipeline")
        with tracer.start_as_current_span("retrieve_all_context") as span:
            t0 = time.time()
            hybrid_task = asyncio.create_task(
                hybrid_search(question, top_k=5, owner_id=owner_id)
            )
            if os.getenv("DISABLE_GRAPH", "0") == "1":
                hybrid_results = await hybrid_task
                graph_context = ""
            else:
                graph_task = asyncio.create_task(
                    retrieve_graph_context(question, owner_id=owner_id)
                )
                hybrid_results, graph_context = await asyncio.gather(hybrid_task, graph_task)

            threshold = float(os.getenv("RERANK_THRESHOLD", "-5.0"))
            pruned_hybrid = prune_irrelevant_context(hybrid_results, threshold=threshold)

            retrieval_ms = (time.time() - t0) * 1000
            span.set_attribute("retrieval.duration_ms", retrieval_ms)
            span.set_attribute("retrieval.hybrid_count", len(pruned_hybrid))
            span.set_attribute("retrieval.has_graph_context", bool(graph_context))
            return pruned_hybrid, graph_context, retrieval_ms

    # ------------------------------------------------------------------ payloads
    @staticmethod
    def _history_turns(chat_history: List[Dict]) -> List[Tuple[str, str]]:
        """(role, wrapped_content) for every valid history turn; shared by all providers."""
        turns = []
        for msg in chat_history or []:
            role = msg.get("role")
            if role not in ("user", "assistant", "model"):
                continue
            tag = "USER" if role == "user" else "ASSISTANT"
            safe = neutralize_markup(str(msg.get("content", "")))
            turns.append(("user" if role == "user" else "assistant",
                          f"<HISTORY_TURN_{tag}>\n{safe}\n</HISTORY_TURN_{tag}>"))
        return turns

    def _build_request(
        self,
        provider: str,
        question: str,
        chat_history: List[Dict],
        hybrid_results: List[Dict],
        graph_context: str,
        stream: bool = False,
    ) -> Tuple[str, dict, dict, List[Dict]]:
        max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))
        system_prompt, included = build_prompt_with_sources(
            hybrid_results, graph_context, max_tokens=max_context_tokens
        )
        api_key = llm_manager.get_api_key(provider)
        safe_question = neutralize_markup(question)
        query_block = f"<USER_QUERY>\n{safe_question}\n</USER_QUERY>"
        settings = llm_manager.generation_settings()
        turns = self._history_turns(chat_history)

        if provider == "gemini":
            model = llm_manager.gemini_model
            if stream:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
            else:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            contents = [
                {"role": "model" if role == "assistant" else "user", "parts": [{"text": text}]}
                for role, text in turns
            ]
            contents.append({"role": "user", "parts": [{"text": query_block}]})
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": contents,
                "generationConfig": {
                    "temperature": settings["temperature"],
                    "maxOutputTokens": settings["max_output_tokens"],
                },
            }
            return url, {"Content-Type": "application/json"}, payload, included

        if provider == "groq":
            messages = [{"role": "system", "content": system_prompt}]
            messages += [{"role": role, "content": text} for role, text in turns]
            messages.append({"role": "user", "content": query_block})
            payload = {
                "model": llm_manager.groq_model,
                "messages": messages,
                "temperature": settings["temperature"],
                "max_tokens": settings["max_output_tokens"],
            }
            if stream:
                payload["stream"] = True
                payload["stream_options"] = {"include_usage": True}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            return "https://api.groq.com/openai/v1/chat/completions", headers, payload, included

        raise ValueError(f"Unknown provider: {provider}")

    def _build_payload(self, provider: str, question: str, chat_history: List[Dict],
                       hybrid_results: List[Dict], graph_context: str) -> tuple[str, dict, dict]:
        """Backward-compatible streaming request builder (url, headers, payload)."""
        url, headers, payload, _ = self._build_request(
            provider, question, chat_history, hybrid_results, graph_context, stream=True
        )
        return url, headers, payload

    async def _on_rate_limit(self, provider: str, api_key: str, attempt: int, max_retries: int) -> bool:
        """Shared 429 policy for ask and ask_stream. Returns True to retry."""
        if provider == "gemini" and llm_manager.rotate_gemini_key(api_key):
            logger.warning(f"429 on attempt {attempt + 1}, rotated Gemini key")
            return True
        llm_manager.switch_to_fallback(provider)
        if attempt < max_retries - 1:
            wait = 2 ** attempt
            logger.warning(f"429 on attempt {attempt + 1}, waiting {wait}s")
            await asyncio.sleep(wait)
            return True
        return False

    # ------------------------------------------------------------------ results
    def _format_citations(self, context_results: List[Dict]) -> List[Dict]:
        """Citations for exactly the excerpts the model saw, numbered like the prompt (R-16)."""
        citations = []
        for idx, r in enumerate(context_results, start=1):
            # Excerpt: the matched child text (what retrieval found), else the parent.
            text = (r.get("text") or r.get("parent_content") or "")[:200]
            meta = r.get("metadata") or {}
            page_number = r.get("page_number") if r.get("page_number") is not None else meta.get("page_number")
            if page_number is None and "page" in meta:
                page_number = meta["page"]
            filename = r.get("filename") or meta.get("filename", "unknown")
            citations.append({
                "index": idx,
                "filename": filename,
                "page_number": page_number,
                "excerpt": text,
                "score": r.get("rerank_score", r.get("score", 0.0)),
            })
        return citations

    async def _log_and_track_metrics(
        self,
        question: str,
        context_results: List[Dict],
        tokens_used: int,
        provider: str = "gemini",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        estimated: bool = True,
    ):
        avg_score = sum(r.get("rerank_score", r.get("score", 0.0)) for r in context_results) / max(len(context_results), 1)
        if input_tokens is not None and output_tokens is not None:
            cost_usd = (input_tokens * _price_per_million(provider, "INPUT")
                        + output_tokens * _price_per_million(provider, "OUTPUT")) / 1_000_000
        else:
            cost_usd = tokens_used * _price_per_million(provider, "OUTPUT") / 1_000_000

        logger.info(
            "RAG Query Executed",
            question_length=len(question),
            question_hash=hashlib.sha256(question.encode()).hexdigest()[:8],
            retrieval_count=len(context_results),
            avg_score=avg_score,
            tokens_used=tokens_used,
            tokens_estimated=estimated,
            provider=provider,
            cost_usd=cost_usd,
        )

        await increment_metric(daily_key("queries_today"), 1, ttl=2 * 86400)
        await increment_metric("total_tokens_used", tokens_used)
        await increment_metric("estimated_cost_usd", cost_usd)
        if estimated:
            await increment_metric("queries_with_estimated_tokens", 1)

    @staticmethod
    def _make_cache_key(
        question: str,
        chat_history: List[Dict],
        tenant_key: str = "",
        owner_id: Optional[str] = None,
        generation: int = 0,
    ) -> str:
        # The rewritten retrieval query is a pure function of (question, history),
        # which are both in the key, so it does not need to be part of it.
        key = owner_id or tenant_key
        try:
            history_str = json.dumps(chat_history, default=str)
        except TypeError:
            history_str = str(chat_history)
        tenant_salt = key or os.getenv("API_KEY", "default")
        gen_part = f"_gen:{generation}" if (key and generation) else ""
        return "query_cache:" + hashlib.sha256(
            f"{tenant_salt}_{question}_{history_str}{gen_part}".encode()
        ).hexdigest()

    async def _prepare(self, question: str, chat_history: List[Dict], owner_id: Optional[str]):
        retrieval_query = await self._rewrite_query(question, chat_history)
        return await self._retrieve_all_context(retrieval_query, owner_id=owner_id)

    @staticmethod
    def _has_context(hybrid_results: List[Dict], graph_context: str) -> bool:
        return bool(hybrid_results) or bool((graph_context or "").strip())

    # ------------------------------------------------------------------ ask
    async def ask(
        self,
        question: str,
        chat_history: List[Dict] = None,
        owner_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        t_start = time.time()
        chat_history = chat_history or []

        generation = await get_cache_generation(owner_id) if owner_id else 0
        cache_key = self._make_cache_key(
            question, chat_history, tenant_key=owner_id or "", owner_id=owner_id, generation=generation
        )
        cached_result = await get_cache(cache_key)
        if cached_result:
            logger.info("Exact-match cache hit", cache_key=cache_key)
            cached_result["timings"]["total_ms"] = (time.time() - t_start) * 1000
            return cached_result

        hybrid_results, graph_context, retrieval_ms = await self._prepare(question, chat_history, owner_id)

        # R-26: nothing relevant retrieved -> refuse without calling the LLM.
        if not self._has_context(hybrid_results, graph_context):
            return {
                "answer": REFUSAL_TEXT,
                "citations": [],
                "tokens_used": 0,
                "timings": {
                    "embedding_ms": 0,
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": 0,
                    "total_ms": (time.time() - t_start) * 1000,
                },
            }

        t1 = time.time()
        answer = ""
        usage: Optional[Tuple[int, int]] = None
        llm_call_succeeded = False
        included: List[Dict] = []
        payload: Dict[str, Any] = {}
        provider = llm_manager.get_active_provider()
        max_retries = _max_retries()

        for attempt in range(max_retries):
            provider = llm_manager.get_active_provider()
            api_key = llm_manager.get_api_key(provider)
            url, headers, payload, included = self._build_request(
                provider, question, chat_history, hybrid_results, graph_context, stream=False
            )
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, headers=headers, json=payload, timeout=60.0)
                    if response.status_code == 429 and await self._on_rate_limit(provider, api_key, attempt, max_retries):
                        continue
                    response.raise_for_status()
                    data = response.json()

                    extracted = ""
                    if provider == "gemini":
                        candidates = data.get("candidates") or []
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                extracted = parts[0].get("text", "")
                        fallback = "No response from Gemini API."
                    else:
                        choices = data.get("choices") or []
                        if choices:
                            extracted = choices[0].get("message", {}).get("content", "")
                        fallback = "Error generating answer. Check server logs."
                    if extracted and extracted.strip():
                        answer = extracted
                        llm_call_succeeded = True
                    else:
                        answer = fallback
                    usage = _usage_from(provider, data)
                    break
            except httpx.HTTPStatusError as e:
                logger.error(f"{provider} API HTTP error", status=e.response.status_code,
                             body=sanitize_error_msg(e.response.text[:300]))
                answer = f"API Error ({e.response.status_code}). Check server logs."
                if e.response.status_code != 429:
                    break
            except Exception as e:
                logger.error(f"{provider} API failed", error_type=e.__class__.__name__)
                answer = "Error generating answer. Check server logs."
                break

        generation_ms = (time.time() - t1) * 1000

        if usage:
            tokens_used = usage[0] + usage[1]
            await self._log_and_track_metrics(question, included, tokens_used, provider,
                                              usage[0], usage[1], estimated=False)
        else:
            tokens_used = (len(json.dumps(payload, default=str)) + len(answer)) // 4
            await self._log_and_track_metrics(question, included, tokens_used, provider, estimated=True)

        result = {
            "answer": answer,
            "citations": self._format_citations(included),
            "tokens_used": tokens_used,
            "tokens_estimated": usage is None,
            "timings": {
                "embedding_ms": 0,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": (time.time() - t_start) * 1000,
            },
        }

        if llm_call_succeeded and answer.strip() and not is_error_placeholder(answer):
            await set_cache(cache_key, result, ttl=86400)

        return result

    # ------------------------------------------------------------------ stream
    async def ask_stream(
        self,
        question: str,
        chat_history: List[Dict] = None,
        owner_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        chat_history = chat_history or []

        generation = await get_cache_generation(owner_id) if owner_id else 0
        cache_key = self._make_cache_key(
            question, chat_history, tenant_key=owner_id or "", owner_id=owner_id, generation=generation
        )
        cached_result = await get_cache(cache_key)
        if cached_result:
            logger.info("Exact-match stream cache hit", cache_key=cache_key)
            words = cached_result["answer"].split(" ")
            for i, word in enumerate(words):
                content = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'text': content})}\n\n"
                await asyncio.sleep(0.01)
            yield f"data: {json.dumps({'citations': cached_result['citations'], 'tokens_used': cached_result['tokens_used']})}\n\n"
            yield "data: [DONE]\n\n"
            return

        hybrid_results, graph_context, _ = await self._prepare(question, chat_history, owner_id)

        # R-26: refuse without an LLM call, as a normal stream.
        if not self._has_context(hybrid_results, graph_context):
            yield f"data: {json.dumps({'text': REFUSAL_TEXT})}\n\n"
            yield f"data: {json.dumps({'citations': [], 'tokens_used': 0})}\n\n"
            yield "data: [DONE]\n\n"
            return

        generated_chars = 0
        full_answer = ""
        stream_succeeded = False
        usage: Optional[Tuple[int, int]] = None
        included: List[Dict] = []
        payload: Dict[str, Any] = {}
        provider = llm_manager.get_active_provider()
        max_retries = _max_retries()

        try:
            for attempt in range(max_retries):
                provider = llm_manager.get_active_provider()
                api_key = llm_manager.get_api_key(provider)
                url, headers, payload, included = self._build_request(
                    provider, question, chat_history, hybrid_results, graph_context, stream=True
                )
                try:
                    async with httpx.AsyncClient() as client:
                        async with client.stream("POST", url, headers=headers, json=payload, timeout=60.0) as response:
                            if response.status_code == 429:
                                if await self._on_rate_limit(provider, api_key, attempt, max_retries):
                                    continue
                                response.raise_for_status()
                            response.raise_for_status()

                            async for line in response.aiter_lines():
                                if not line.startswith("data: "):
                                    continue
                                data_str = line[6:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                except ValueError:
                                    continue  # partial / non-JSON SSE line
                                usage = _usage_from(provider, chunk) or usage
                                content = ""
                                if provider == "gemini":
                                    parts = (chunk.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
                                    if parts:
                                        content = parts[0].get("text", "")
                                else:
                                    choices = chunk.get("choices") or []
                                    if choices:
                                        content = (choices[0].get("delta") or {}).get("content") or ""
                                if content:
                                    generated_chars += len(content)
                                    full_answer += content
                                    yield f"data: {json.dumps({'text': content})}\n\n"

                            if full_answer.strip():
                                stream_succeeded = True
                            break  # clean exit, do not retry

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt < max_retries - 1:
                        if await self._on_rate_limit(provider, api_key, attempt, max_retries):
                            continue
                    raise

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                msg = "\n\n⚠️ **API Rate Limit Exceeded.** You've hit the rate limits for all fallback providers."
            else:
                logger.error("API stream HTTP error", status=e.response.status_code)
                msg = f"Stream error ({e.response.status_code}). Check server logs."
            yield f"data: {json.dumps({'text': msg})}\n\n"
        except Exception as e:
            logger.error("API stream failed", error_type=e.__class__.__name__)
            yield f"data: {json.dumps({'text': 'Streaming error occurred. Check server logs.'})}\n\n"

        if usage:
            tokens_used = usage[0] + usage[1]
            await self._log_and_track_metrics(question, included, tokens_used, provider,
                                              usage[0], usage[1], estimated=False)
        else:
            tokens_used = (len(json.dumps(payload, default=str)) + generated_chars) // 4
            await self._log_and_track_metrics(question, included, tokens_used, provider, estimated=True)

        citations = self._format_citations(included)
        yield f"data: {json.dumps({'citations': citations, 'tokens_used': tokens_used})}\n\n"
        yield "data: [DONE]\n\n"

        if stream_succeeded and not is_error_placeholder(full_answer):
            result = {
                "answer": full_answer,
                "citations": citations,
                "tokens_used": tokens_used,
                "timings": {"embedding_ms": 0, "retrieval_ms": 0, "generation_ms": 0, "total_ms": 0},
            }
            await set_cache(cache_key, result, ttl=86400)


pipeline = RAGPipeline()
