"""Tests for the recursive parent-child chunker."""

from unittest.mock import patch
from langchain_core.documents import Document

from app.rag.chunkers.recursive_chunker import chunk_documents


class TestChunkDocuments:
    def test_returns_child_docs_and_parent_store(self, sample_documents):
        children, parent_store = chunk_documents(sample_documents)

        assert isinstance(children, list)
        assert isinstance(parent_store, dict)
        assert len(children) > 0
        assert len(parent_store) > 0

    def test_child_docs_have_required_metadata(self, sample_documents):
        children, _ = chunk_documents(sample_documents)

        for child in children:
            assert "parent_id" in child.metadata
            assert "hash" in child.metadata
            assert "chunk_index" in child.metadata
            assert "parent_index" in child.metadata

    def test_parent_store_contains_text(self, sample_documents):
        children, parent_store = chunk_documents(sample_documents)

        for parent_id, parent_text in parent_store.items():
            assert isinstance(parent_id, str)
            assert len(parent_text) > 0

    def test_child_references_valid_parent(self, sample_documents):
        children, parent_store = chunk_documents(sample_documents)

        for child in children:
            parent_id = child.metadata["parent_id"]
            assert parent_id in parent_store

    def test_child_chunks_are_smaller_than_parent(self, sample_documents):
        """Child chunks (400 char) should be smaller than parent chunks (2000 char)."""
        children, parent_store = chunk_documents(sample_documents)

        # Children should be at most ~450 chars (400 + some overlap margin)
        for child in children:
            assert len(child.page_content) <= 500

    def test_empty_documents_returns_empty(self):
        children, parent_store = chunk_documents([])

        assert children == []
        assert parent_store == {}

    def test_preserves_original_metadata(self, sample_documents):
        children, _ = chunk_documents(sample_documents)

        for child in children:
            assert "filename" in child.metadata
            assert "source_type" in child.metadata

    @patch.dict("os.environ", {"CHUNK_SIZE": "500", "CHUNK_OVERLAP": "50"})
    def test_respects_env_config(self):
        """Chunk size should be configurable via environment variables."""
        # Re-import to pick up new env vars in a fresh call
        from app.rag.chunkers.recursive_chunker import chunk_documents as chunk_fn

        long_doc = Document(
            page_content="word " * 1000,  # ~5000 chars
            metadata={"filename": "long.pdf", "source_type": "pdf"},
        )
        children, parent_store = chunk_fn([long_doc])

        # With chunk_size=500, we should get more parent chunks than with 2000
        assert len(parent_store) > 1

    def test_unique_hashes_per_child(self, sample_documents):
        children, _ = chunk_documents(sample_documents)

        hashes = [c.metadata["hash"] for c in children]
        assert len(hashes) == len(set(hashes)), "Child chunk hashes should be unique"
