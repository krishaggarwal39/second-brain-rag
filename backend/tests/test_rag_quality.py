"""Batch 6: R-15, R-16, R-26, R-27, R-28, R-29, R-48 (prompt/pipeline parts)."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.rag.context_engineering import (
    REFUSAL_TEXT,
    build_prompt_with_sources,
    dedupe_by_parent,
    neutralize_markup,
)
from app.rag.pipeline import RAGPipeline, _max_retries, _price_per_million, _usage_from


@pytest.fixture
def pipeline():
    return RAGPipeline()


def _chunk(text, parent_id=None, parent=None, score=1.0, fn="a.txt"):
    d = {"text": text, "filename": fn, "rerank_score": score}
    if parent_id:
        d["parent_id"] = parent_id
    if parent:
        d["parent_content"] = parent
    return d


# ------------------------------------------------------------ R-15
def test_dedupe_keeps_best_child_per_parent():
    res = [
        _chunk("c1", "p1", "PARENT ONE", 3.0),
        _chunk("c2", "p1", "PARENT ONE", 2.0),
        _chunk("c3", "p2", "PARENT TWO", 1.0),
        _chunk("orphan"),
    ]
    out = dedupe_by_parent(res)
    assert [d["text"] for d in out] == ["c1", "c3", "orphan"]


def test_prompt_contains_parent_text_once():
    res = [_chunk("c1", "p1", "SHARED PARENT"), _chunk("c2", "p1", "SHARED PARENT")]
    prompt, included = build_prompt_with_sources(res, "", 4000)
    assert prompt.count("SHARED PARENT") == 1
    assert len(included) == 1


# ------------------------------------------------------------ R-16
def test_citations_match_included_chunks_only(pipeline):
    big = "word " * 900
    res = [_chunk("first", parent=big), _chunk("second", parent=big), _chunk("third", parent=big)]
    prompt, included = build_prompt_with_sources(res, "", max_tokens=1500)
    assert 0 < len(included) < 3
    cites = pipeline._format_citations(included)
    assert [c["index"] for c in cites] == list(range(1, len(included) + 1))
    for c in cites:
        assert f"Excerpt [{c['index']}]" in prompt
    assert cites[0]["excerpt"] == "first"  # child text, not parent[:200]


def test_prompt_asks_for_bracket_citations_and_no_outside_knowledge():
    prompt, _ = build_prompt_with_sources([_chunk("x")], "", 4000)
    assert "[1]" in prompt
    assert "general knowledge" not in prompt.lower()
    assert "Do not use outside knowledge" in prompt
    assert REFUSAL_TEXT in prompt


# ------------------------------------------------------------ R-48 escaping
def test_neutralize_keeps_ampersand_and_quotes():
    txt = 'R&D spent "40%" on Tom\'s </CONTEXT_x> tag'
    out = neutralize_markup(txt)
    assert "R&D" in out and '"40%"' in out and "Tom's" in out
    assert "</CONTEXT_x>" not in out


def test_max_retries_zero_is_safe(monkeypatch):
    monkeypatch.setenv("MAX_API_RETRIES", "0")
    assert _max_retries() == 1
    monkeypatch.setenv("MAX_API_RETRIES", "nope")
    assert _max_retries() == 3


# ------------------------------------------------------------ R-26
@pytest.mark.asyncio
async def test_ask_refuses_without_llm_when_no_context(pipeline):
    with patch.object(pipeline, "_retrieve_all_context", new_callable=AsyncMock, return_value=([], "", 1.0)), \
         patch("app.rag.pipeline.get_cache", new_callable=AsyncMock, return_value=None), \
         patch("app.rag.pipeline.set_cache", new_callable=AsyncMock) as set_cache, \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        res = await pipeline.ask("anything?", owner_id="u1")
    assert res["answer"] == REFUSAL_TEXT and res["citations"] == []
    post.assert_not_called()
    set_cache.assert_not_called()


@pytest.mark.asyncio
async def test_stream_refuses_without_llm_when_no_context(pipeline):
    with patch.object(pipeline, "_retrieve_all_context", new_callable=AsyncMock, return_value=([], "", 1.0)), \
         patch("app.rag.pipeline.get_cache", new_callable=AsyncMock, return_value=None), \
         patch("httpx.AsyncClient.stream") as stream:
        chunks = [c async for c in pipeline.ask_stream("anything?", owner_id="u1")]
    stream.assert_not_called()
    assert REFUSAL_TEXT in chunks[0]
    assert chunks[-1] == "data: [DONE]\n\n"
    assert json.loads(chunks[1][6:])["citations"] == []


def test_generation_settings_in_payloads(pipeline, monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "321")
    _, _, gem, _ = pipeline._build_request("gemini", "q", [], [_chunk("x")], "")
    assert gem["generationConfig"] == {"temperature": 0.1, "maxOutputTokens": 321}
    _, _, groq, _ = pipeline._build_request("groq", "q", [], [_chunk("x")], "", stream=True)
    assert groq["temperature"] == 0.1 and groq["max_tokens"] == 321
    assert groq["stream_options"] == {"include_usage": True}


def test_non_stream_gemini_url_uses_generate_content(pipeline):
    url, _, _, _ = pipeline._build_request("gemini", "q", [], [_chunk("x")], "", stream=False)
    assert ":generateContent?" in url


# ------------------------------------------------------------ R-27
def test_usage_parsing():
    assert _usage_from("gemini", {"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}}) == (7, 3)
    assert _usage_from("groq", {"usage": {"prompt_tokens": 5, "completion_tokens": 2}}) == (5, 2)
    assert _usage_from("groq", {"x_groq": {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}}) == (1, 1)
    assert _usage_from("groq", {}) is None


def test_price_configurable(monkeypatch):
    monkeypatch.delenv("COST_PER_1M_INPUT_GROQ", raising=False)
    monkeypatch.setenv("COST_PER_1M_TOKENS", "2.0")
    assert _price_per_million("groq", "INPUT") == 2.0
    monkeypatch.setenv("COST_PER_1M_INPUT_GROQ", "0.59")
    assert _price_per_million("groq", "INPUT") == 0.59


@pytest.mark.asyncio
async def test_ask_uses_provider_usage(pipeline):
    resp = MagicMock(status_code=200)
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": "answer [1]"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }
    with patch.object(pipeline, "_retrieve_all_context", new_callable=AsyncMock,
                      return_value=([_chunk("x")], "", 1.0)), \
         patch("app.rag.pipeline.get_cache", new_callable=AsyncMock, return_value=None), \
         patch("app.rag.pipeline.set_cache", new_callable=AsyncMock), \
         patch("app.rag.pipeline.increment_metric", new_callable=AsyncMock) as inc, \
         patch("app.rag.pipeline.llm_manager.get_active_provider", return_value="groq"), \
         patch("app.rag.pipeline.llm_manager.get_api_key", return_value="k"), \
         patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=resp):
        res = await pipeline.ask("q", owner_id="u1")
    assert res["tokens_used"] == 120 and res["tokens_estimated"] is False
    assert not any(c.args[0] == "queries_with_estimated_tokens" for c in inc.call_args_list)


# ------------------------------------------------------------ R-29
@pytest.mark.asyncio
async def test_query_rewrite_used_for_retrieval_only(pipeline, monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE", "true")
    history = [{"role": "user", "content": "Tell me about Project Falcon"},
               {"role": "assistant", "content": "It is an acquisition."}]
    with patch("app.rag.pipeline.llm_manager.generate", new_callable=AsyncMock,
               return_value="Project Falcon budget\n"), \
         patch.object(pipeline, "_retrieve_all_context", new_callable=AsyncMock,
                      return_value=([], "", 1.0)) as retrieve, \
         patch("app.rag.pipeline.get_cache", new_callable=AsyncMock, return_value=None):
        await pipeline.ask("what was its budget?", history, owner_id="u1")
    assert retrieve.call_args.args[0] == "Project Falcon budget"


@pytest.mark.asyncio
async def test_query_rewrite_falls_back(pipeline, monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE", "true")
    with patch("app.rag.pipeline.llm_manager.generate", new_callable=AsyncMock, side_effect=RuntimeError):
        assert await pipeline._rewrite_query("q?", [{"role": "user", "content": "x"}]) == "q?"
    with patch("app.rag.pipeline.llm_manager.generate", new_callable=AsyncMock, return_value=None):
        assert await pipeline._rewrite_query("q?", [{"role": "user", "content": "x"}]) == "q?"
    # no history -> no LLM call
    with patch("app.rag.pipeline.llm_manager.generate", new_callable=AsyncMock) as gen:
        assert await pipeline._rewrite_query("q?", []) == "q?"
        gen.assert_not_called()
    monkeypatch.setenv("QUERY_REWRITE", "false")
    with patch("app.rag.pipeline.llm_manager.generate", new_callable=AsyncMock) as gen:
        assert await pipeline._rewrite_query("q?", [{"role": "user", "content": "x"}]) == "q?"
        gen.assert_not_called()


# ------------------------------------------------------------ R-28
def test_concurrent_429_rotates_once(monkeypatch):
    from app.core.llm_manager import LLMManager

    monkeypatch.setenv("GOOGLE_API_KEY", "k1,k2,k3")
    m = LLMManager()
    # Two requests both used k1 and both got 429
    assert m.rotate_gemini_key("k1") is True
    assert m.gemini_key == "k2"
    assert m.rotate_gemini_key("k1") is True  # stale report: no second rotation
    assert m.gemini_key == "k2"


def test_rotation_recovers_after_cooldown(monkeypatch):
    from app.core import llm_manager as mod

    monkeypatch.setenv("GOOGLE_API_KEY", "k1,k2")
    monkeypatch.setenv("GEMINI_KEY_COOLDOWN_SECONDS", "60")
    m = mod.LLMManager()
    now = [1000.0]
    monkeypatch.setattr(mod.time, "time", lambda: now[0])
    assert m.rotate_gemini_key("k1") is True and m.gemini_key == "k2"
    assert m.rotate_gemini_key("k2") is False  # both cooling down -> fail over
    now[0] += 61
    assert m.rotate_gemini_key("k2") is True and m.gemini_key == "k1"  # k1 is back


def test_rotation_thread_safety(monkeypatch):
    from app.core.llm_manager import LLMManager
    import threading

    monkeypatch.setenv("GOOGLE_API_KEY", "a,b,c,d")
    m = LLMManager()
    threads = [threading.Thread(target=m.rotate_gemini_key, args=("a",)) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.gemini_key == "b"


def test_asyncio_import_used():
    assert asyncio  # keep import for pytest-asyncio collection parity
