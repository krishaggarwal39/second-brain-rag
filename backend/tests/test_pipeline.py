import pytest
from app.rag.pipeline import RAGPipeline

@pytest.fixture
def pipeline():
    return RAGPipeline()

def test_make_cache_key(pipeline, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-tenant")
    key1 = pipeline._make_cache_key("hello world", [{"role": "user", "content": "hi"}])
    key2 = pipeline._make_cache_key("hello world", [{"role": "user", "content": "hi"}])
    key3 = pipeline._make_cache_key("hello world", [{"role": "user", "content": "bye"}])
    
    assert key1 == key2
    assert key1 != key3
    
    monkeypatch.setenv("API_KEY", "another-tenant")
    key4 = pipeline._make_cache_key("hello world", [{"role": "user", "content": "hi"}])
    assert key1 != key4

    # Cache key with owner_id cross-user isolation
    k_owner1 = pipeline._make_cache_key("hello world", [{"role": "user", "content": "hi"}], owner_id="1")
    k_owner2 = pipeline._make_cache_key("hello world", [{"role": "user", "content": "hi"}], owner_id="2")
    assert k_owner1 != k_owner2
    assert k_owner1 != key1

    # Passing via tenant_key parameter gives equivalent isolation
    k_tenant1 = pipeline._make_cache_key("hello world", [{"role": "user", "content": "hi"}], tenant_key="1")
    assert k_tenant1 == k_owner1

def test_build_payload_gemini(pipeline):
    url, headers, payload = pipeline._build_payload(
        provider="gemini",
        question="What is the matrix? <hack>",
        chat_history=[{"role": "user", "content": "hello"}],
        hybrid_results=[{"text": "context 1", "score": 0.9}],
        graph_context="Graph data"
    )
    
    assert "generativelanguage.googleapis.com" in url
    assert headers["Content-Type"] == "application/json"
    
    # Check escaping
    assert "<hack>" not in str(payload)
    assert "&lt;hack&gt;" in str(payload)
    
    assert len(payload["contents"]) == 2
    assert payload["contents"][1]["role"] == "user"

def test_build_payload_groq(pipeline):
    url, headers, payload = pipeline._build_payload(
        provider="groq",
        question="Who are you? <hack>",
        chat_history=[{"role": "user", "content": "hello"}],
        hybrid_results=[],
        graph_context=""
    )
    
    assert "api.groq.com" in url
    assert headers["Content-Type"] == "application/json"
    
    assert "<hack>" not in str(payload)
    assert "&lt;hack&gt;" in str(payload)
    
    assert payload["model"] == "llama-3.3-70b-versatile"
    assert len(payload["messages"]) == 3 # System + History + Question

def test_format_citations(pipeline):
    results = [
        {"filename": "doc1.pdf", "text": "A long text " * 100, "rerank_score": 0.95, "metadata": {"page_number": 4}},
        {"text": "No filename", "score": 0.8},
        {"text": "Doc with top-level page", "score": 0.9, "page_number": 12, "filename": "doc2.pdf"},
    ]
    citations = pipeline._format_citations(results)
    
    assert len(citations) == 3
    assert citations[0]["filename"] == "doc1.pdf"
    assert citations[0]["score"] == 0.95
    assert citations[0]["page_number"] == 4
    assert len(citations[0]["excerpt"]) == 200 # truncated
    
    assert citations[1]["filename"] == "unknown"
    assert citations[1]["score"] == 0.8
    assert citations[1]["page_number"] is None

    assert citations[2]["filename"] == "doc2.pdf"
    assert citations[2]["score"] == 0.9
    assert citations[2]["page_number"] == 12


def test_make_cache_key_with_generation(pipeline):
    history = [{"role": "user", "content": "hi"}]
    # Owner 1 at gen 0
    k_gen0 = pipeline._make_cache_key("hello", history, owner_id="1", generation=0)
    # Owner 1 at gen 1 (bumped)
    k_gen1 = pipeline._make_cache_key("hello", history, owner_id="1", generation=1)
    # Owner 1 at gen 2 (bumped again)
    k_gen2 = pipeline._make_cache_key("hello", history, owner_id="1", generation=2)

    assert k_gen0 != k_gen1
    assert k_gen1 != k_gen2

    # User 2 at gen 1 is distinct from User 1 at gen 1
    k_user2_gen1 = pipeline._make_cache_key("hello", history, owner_id="2", generation=1)
    assert k_gen1 != k_user2_gen1

    # owner_id=None falls back to current behavior regardless of generation
    k_none_g0 = pipeline._make_cache_key("hello", history, owner_id=None, generation=0)
    k_none_g1 = pipeline._make_cache_key("hello", history, owner_id=None, generation=1)
    assert k_none_g0 == k_none_g1


@pytest.mark.asyncio
async def test_ask_caching_with_success_flag(pipeline):
    from unittest.mock import AsyncMock, patch, MagicMock

    with patch.object(pipeline, "_retrieve_all_context", new_callable=AsyncMock) as mock_retrieve, \
         patch("app.rag.pipeline.set_cache", new_callable=AsyncMock) as mock_set_cache, \
         patch("app.rag.pipeline.get_cache", new_callable=AsyncMock, return_value=None), \
         patch("app.rag.pipeline.increment_metric", new_callable=AsyncMock), \
         patch("app.rag.pipeline.llm_manager.get_active_provider", return_value="groq"), \
         patch("app.rag.pipeline.llm_manager.get_api_key", return_value="fake-key"):

        mock_retrieve.return_value = ([], "", 5.0)

        # 1. Legit answer containing the word "Error" (previously false-positive: was blocked from cache)
        mock_response_ok = MagicMock()
        mock_response_ok.status_code = 200
        mock_response_ok.raise_for_status = MagicMock()
        mock_response_ok.json.return_value = {
            "choices": [{"message": {"content": "The ValueError was raised because the input was invalid."}}],
            "usage": {"total_tokens": 10},
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response_ok):
            res = await pipeline.ask("How to fix my code?", owner_id="user1")
            assert "ValueError" in res["answer"]
            mock_set_cache.assert_called_once()
            _, set_cache_args, _ = mock_set_cache.mock_calls[0]
            assert set_cache_args[1]["answer"] == "The ValueError was raised because the input was invalid."

        mock_set_cache.reset_mock()

        # 2. HTTP error (e.g. 500) returns error string and must NOT be cached
        import httpx
        req = httpx.Request("POST", "http://test")
        resp_500 = httpx.Response(500, request=req, text="Server Error")
        http_err = httpx.HTTPStatusError("500 Server Error", request=req, response=resp_500)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=http_err):
            res = await pipeline.ask("Question causing error?", owner_id="user1")
            assert "API Error (500)" in res["answer"]
            mock_set_cache.assert_not_called()

        mock_set_cache.reset_mock()

        # 3. Model returning empty content / error placeholder must NOT be cached
        mock_response_empty = MagicMock()
        mock_response_empty.status_code = 200
        mock_response_empty.raise_for_status = MagicMock()
        mock_response_empty.json.return_value = {
            "choices": [{"message": {"content": ""}}],
            "usage": {"total_tokens": 0},
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response_empty):
            res = await pipeline.ask("Question with empty answer?", owner_id="user1")
            mock_set_cache.assert_not_called()


@pytest.mark.asyncio
async def test_ask_stream_caching_validation(pipeline):
    from unittest.mock import AsyncMock, patch

    with patch.object(pipeline, "_retrieve_all_context", new_callable=AsyncMock) as mock_retrieve, \
         patch("app.rag.pipeline.set_cache", new_callable=AsyncMock) as mock_set_cache, \
         patch("app.rag.pipeline.get_cache", new_callable=AsyncMock, return_value=None), \
         patch("app.rag.pipeline.increment_metric", new_callable=AsyncMock), \
         patch("app.rag.pipeline.llm_manager.get_active_provider", return_value="groq"), \
         patch("app.rag.pipeline.llm_manager.get_api_key", return_value="fake-key"):

        mock_retrieve.return_value = ([], "", 5.0)

        # 1. Successful stream with non-empty answer is cached
        class MockStreamSuccess:
            status_code = 200
            def raise_for_status(self): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def aiter_lines(self):
                yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
                yield 'data: {"choices": [{"delta": {"content": " world"}}]}'
                yield 'data: [DONE]'

        with patch("httpx.AsyncClient.stream", return_value=MockStreamSuccess()):
            chunks = [c async for c in pipeline.ask_stream("Hi", owner_id="user1")]
            assert len(chunks) > 0
            mock_set_cache.assert_called_once()
            _, args, _ = mock_set_cache.mock_calls[0]
            assert args[1]["answer"] == "Hello world"

        mock_set_cache.reset_mock()

        # 2. Stream with empty content is NOT cached
        class MockStreamEmpty:
            status_code = 200
            def raise_for_status(self): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def aiter_lines(self):
                yield 'data: [DONE]'

        with patch("httpx.AsyncClient.stream", return_value=MockStreamEmpty()):
            chunks = [c async for c in pipeline.ask_stream("Hi", owner_id="user1")]
            mock_set_cache.assert_not_called()

