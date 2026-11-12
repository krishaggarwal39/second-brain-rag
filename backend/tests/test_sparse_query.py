"""R-22 / R-24: sparse query sanitisation and identical semantics on both backends."""
import os
import uuid

import pytest

from app.rag.vectorstore.bm25_store import (
    BM25Store,
    PostgresBM25Store,
    fts5_or_query,
    query_tokens,
    tsquery_or_query,
)

DATABASE_URL = os.getenv("DATABASE_URL")
requires_postgres = pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")

TRICKY = [
    "What is NOT allowed",
    "cats AND dogs",
    'say "hello" world',
    "prefix* col:value -minus (paren) NEAR(a b)",
    "OR",
]


def test_tokens_strip_operators_and_punctuation():
    assert query_tokens('a "b" c* d:e -f (g)') == ["a", "b", "c", "d", "e", "f", "g"]
    assert query_tokens("?!...  ,,") == []
    assert query_tokens("snake_case") == ["snake", "case"]


def test_fts5_query_quotes_every_token():
    assert fts5_or_query("cats AND dogs") == '"cats" OR "AND" OR "dogs"'
    assert fts5_or_query("!!!") == ""


def test_tsquery_or_join():
    assert tsquery_or_query("long question about budgets") == "long | question | about | budgets"


def test_token_cap():
    assert len(query_tokens(" ".join(["w"] * 500))) == 64


def _seed(store, owner):
    doc = f"sq_{uuid.uuid4().hex[:8]}"
    rows = [
        {"id": f"{doc}_1", "doc_id": doc, "owner_id": owner,
         "text": "Smoking is not allowed inside the building."},
        {"id": f"{doc}_2", "doc_id": doc, "owner_id": owner,
         "text": "Cats and dogs are welcome in the garden."},
        {"id": f"{doc}_3", "doc_id": doc, "owner_id": owner,
         "text": "The quarterly budget for Project Falcon was approved."},
    ]
    store.add_documents(rows)
    return doc


def _check_semantics(store, owner):
    # Operators used to raise "fts5: syntax error" and return [].
    for q in TRICKY[:-1]:
        store.search(q, top_k=3, owner_id=owner)  # must not raise
    assert store.search("What is NOT allowed", top_k=3, owner_id=owner)
    assert store.search("cats AND dogs", top_k=3, owner_id=owner)
    # Long question with extra words must still match (OR, not AND).
    hits = store.search(
        "Could you please tell me what the quarterly budget of Falcon looked like overall?",
        top_k=3, owner_id=owner,
    )
    assert any("budget" in h["text"] for h in hits)
    assert store.search("?!...", top_k=3, owner_id=owner) == []
    assert store.search("cats", top_k=3, owner_id="someone-else") == []


def test_sqlite_semantics():
    store = BM25Store()
    owner = f"o_{uuid.uuid4().hex[:6]}"
    doc = _seed(store, owner)
    try:
        _check_semantics(store, owner)
    finally:
        store.delete_documents_by_doc_id(doc, owner_id=owner)


@requires_postgres
def test_postgres_semantics():
    store = PostgresBM25Store(database_url=DATABASE_URL)
    owner = f"o_{uuid.uuid4().hex[:6]}"
    doc = _seed(store, owner)
    try:
        _check_semantics(store, owner)
    finally:
        store.delete_documents_by_doc_id(doc, owner_id=owner)
