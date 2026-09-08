import pytest
from unittest.mock import patch, MagicMock
import chromadb

from app.rag.vectorstore.chroma_store import add_documents, query, delete_document, list_documents, get_stats

@pytest.fixture
def mock_chroma_client():
    client = chromadb.EphemeralClient()
    with patch("app.rag.vectorstore.chroma_store.get_client", return_value=client):
        yield client
        
@pytest.mark.asyncio
async def test_chroma_store_cycle(mock_chroma_client):
    # 1. Add Documents
    docs = ["Hello world", "This is a test"]
    embeddings = [[0.1, 0.1], [0.2, 0.2]]
    metadatas = [
        {"hash": "hash1", "doc_id": "doc1", "filename": "file1.txt"},
        {"hash": "hash2", "doc_id": "doc1", "filename": "file1.txt"}
    ]
    
    await add_documents(docs, embeddings, metadatas)
    
    # Verify stats
    stats = get_stats()
    assert stats["total_chunks"] == 2
    # 2. Add same documents again (should skip because of same hash/doc_id combination)
    await add_documents(docs, embeddings, metadatas)
    stats = get_stats()
    assert stats["total_chunks"] == 2
    
    # 3. Query
    results = await query([0.1, 0.1], n_results=1, score_threshold=0.5)
    assert len(results) == 1
    assert results[0]["text"] == "Hello world"
    assert results[0]["score"] > 0.9
    
    # 4. List Documents
    doc_list = await list_documents()
    assert len(doc_list) == 1
    assert doc_list[0]["doc_id"] == "doc1"
    assert doc_list[0]["chunk_count"] == 2
    
    # 5. Delete Document
    await delete_document("doc1")
    stats_after = get_stats()
    assert stats_after["total_chunks"] == 0


@pytest.mark.asyncio
async def test_chroma_delete_document_with_owner_id(mock_chroma_client):
    """Test delete_document with owner_id only deletes chunks of that owner."""
    docs = ["User 1 doc", "User 2 doc"]
    embeddings = [[0.1, 0.1], [0.2, 0.2]]
    metadatas = [
        {"hash": "h_u1", "doc_id": "shared_doc", "owner_id": "user1", "filename": "f.txt"},
        {"hash": "h_u2", "doc_id": "shared_doc", "owner_id": "user2", "filename": "f.txt"},
    ]
    await add_documents(docs, embeddings, metadatas)
    assert get_stats()["total_chunks"] == 2

    # Deleting as user1 should delete only user1's chunk
    await delete_document("shared_doc", owner_id="user1")
    assert get_stats()["total_chunks"] == 1

    # Remaining doc should belong to user2
    remaining = await query([0.2, 0.2], n_results=5, score_threshold=0.0)
    assert len(remaining) == 1
    assert remaining[0]["metadata"]["owner_id"] == "user2"

    # Clean up user2's doc
    await delete_document("shared_doc", owner_id="user2")
    assert get_stats()["total_chunks"] == 0

@pytest.mark.asyncio
async def test_local_embedder():
    from app.rag.embeddings.local_embedder import embed_documents
    with patch("app.rag.embeddings.local_embedder._get_model") as mock_get_model:
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1, 0.2], [0.3, 0.4]])
        mock_get_model.return_value = mock_model
        
        texts = ["Text 1", "Text 2"]
        result1 = await embed_documents(texts)
        assert len(result1) == 2
        assert result1[0] == [0.1, 0.2]
