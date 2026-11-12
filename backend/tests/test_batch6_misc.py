"""Batch 6 leftovers: PDF tables (R-48), Chroma fallback flag and non-blocking calls (R-25)."""
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.rag.loaders.pdf_loader import _table_to_text


def test_pdf_table_rendered_as_rows():
    table = [["Quarter", "Revenue"], ["Q1", "1,200"], [None, "  multi\nline "], [None, None]]
    assert _table_to_text(table) == "Quarter | Revenue\nQ1 | 1,200\n | multi line"


def test_chroma_fallback_is_opt_in(monkeypatch):
    from app.rag.vectorstore import chroma_store

    monkeypatch.setattr(chroma_store, "_client", None)
    monkeypatch.setenv("CHROMA_ALLOW_LOCAL_FALLBACK", "false")
    with patch.object(chroma_store.chromadb, "HttpClient", side_effect=ConnectionError("down")), \
         patch.object(chroma_store.chromadb, "PersistentClient") as local:
        with pytest.raises(ConnectionError):
            chroma_store.get_client()
        local.assert_not_called()
    monkeypatch.setattr(chroma_store, "_client", None)


def test_chroma_fallback_when_enabled(monkeypatch, tmp_path):
    from app.rag.vectorstore import chroma_store

    monkeypatch.setattr(chroma_store, "_client", None)
    monkeypatch.setenv("CHROMA_ALLOW_LOCAL_FALLBACK", "true")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path))
    sentinel = object()
    with patch.object(chroma_store.chromadb, "HttpClient", side_effect=ConnectionError("down")), \
         patch.object(chroma_store.chromadb, "PersistentClient", return_value=sentinel):
        assert chroma_store.get_client() is sentinel
    monkeypatch.setattr(chroma_store, "_client", None)


@pytest.mark.asyncio
async def test_chroma_query_does_not_block_event_loop():
    """A slow sync Chroma call must not freeze other coroutines (R-25)."""
    import time
    from app.rag.vectorstore import chroma_store

    collection = MagicMock()

    def slow_query(**kwargs):
        time.sleep(0.3)
        return {"documents": [["d"]], "metadatas": [[{}]], "distances": [[0.0]]}

    collection.query.side_effect = slow_query
    ticks = []

    async def ticker():
        for _ in range(5):
            ticks.append(time.perf_counter())
            await asyncio.sleep(0.03)

    with patch.object(chroma_store, "get_collection", return_value=collection):
        res, _ = await asyncio.gather(chroma_store.query([0.1]), ticker())
    assert res[0]["text"] == "d"
    assert len(ticks) == 5 and ticks[-1] - ticks[0] < 0.25
