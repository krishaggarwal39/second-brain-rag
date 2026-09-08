import os
import uuid
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.logging import get_logger

logger = get_logger(__name__)

_client = None

# Retry decorator for ChromaDB network operations
_chroma_retry = retry(
    stop=stop_after_attempt(int(os.getenv("DB_MAX_RETRIES", "3"))),
    wait=wait_exponential(multiplier=1, min=0.5, max=5.0),
    retry=retry_if_exception_type((ConnectionError, OSError, TimeoutError)),
    reraise=True,
)


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is not None:
        return _client

    chroma_host = os.getenv("CHROMA_HOST", "chroma")
    chroma_port = int(os.getenv("CHROMA_PORT", "8001"))
    chroma_token = os.getenv("CHROMA_AUTH_TOKEN")

    try:
        if chroma_token:
            settings = Settings(
                anonymized_telemetry=False,
                chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                chroma_client_auth_credentials=chroma_token,
            )
        else:
            settings = Settings(anonymized_telemetry=False)

        _client = chromadb.HttpClient(host=chroma_host, port=chroma_port, settings=settings)
    except Exception:
        # Fallback to local persistent client (useful when running outside Docker)
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
        _client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

    return _client


@_chroma_retry
def get_collection():
    return get_client().get_or_create_collection("rag_collection")


@_chroma_retry
async def add_documents(
    docs: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
):
    collection = get_collection()

    ids = [meta.get("hash", str(uuid.uuid4())) for meta in metadatas]

    existing_set = set(collection.get(ids=ids)["ids"])

    new_docs, new_embs, new_metas, new_ids = [], [], [], []
    for i, doc_id in enumerate(ids):
        if doc_id not in existing_set:
            new_docs.append(docs[i])
            new_embs.append(embeddings[i])
            new_metas.append(metadatas[i])
            new_ids.append(doc_id)

    if not new_ids:
        return

    clean_metas = []
    for m in new_metas:
        clean = {}
        for k, v in m.items():
            # ChromaDB only accepts scalar metadata values
            clean[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        clean.setdefault("doc_id", clean.get("filename", clean.get("url", "unknown")))
        clean_metas.append(clean)

    collection.add(
        ids=new_ids,
        embeddings=new_embs,
        documents=new_docs,
        metadatas=clean_metas,
    )


@_chroma_retry
async def query(
    embedding: List[float],
    n_results: int = 5,
    score_threshold: float = 0.0,
    filters: Optional[Dict] = None,
) -> List[Dict]:
    collection = get_collection()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where=filters,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if not results["documents"] or not results["documents"][0]:
        return output

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB returns L2 distances; convert to a 0-1 similarity score.
        score = 1.0 / (1.0 + dist)
        if score >= score_threshold:
            output.append({"text": doc, "metadata": meta, "score": score})

    return output


@_chroma_retry
async def delete_document(doc_id: str, owner_id: Optional[str] = None):
    if owner_id:
        where_clause = {"$and": [{"doc_id": doc_id}, {"owner_id": str(owner_id)}]}
    else:
        where_clause = {"doc_id": doc_id}
    get_collection().delete(where=where_clause)


@_chroma_retry
async def get_document_parents(doc_id: str, owner_id: Optional[str] = None) -> set:
    collection = get_collection()
    if owner_id:
        where_clause = {"$and": [{"doc_id": doc_id}, {"owner_id": str(owner_id)}]}
    else:
        where_clause = {"doc_id": doc_id}
    results = collection.get(where=where_clause, include=["metadatas"])
    parents = set()
    for m in results.get("metadatas", []):
        if m and "parent_id" in m:
            parents.add(m["parent_id"])
    return parents


@_chroma_retry
async def get_chunk_metadatas(ids: List[str]) -> Dict[str, Dict]:
    """Retrieve metadatas for a list of chunk IDs from ChromaDB."""
    if not ids:
        return {}
    try:
        collection = get_collection()
        results = collection.get(ids=ids, include=["metadatas"])
        res_ids = results.get("ids", [])
        res_metas = results.get("metadatas", [])
        return {cid: meta for cid, meta in zip(res_ids, res_metas) if meta}
    except Exception as e:
        logger.warning(f"Failed to fetch chunk metadatas from Chroma: {e}")
        return {}


@_chroma_retry
async def list_documents() -> List[Dict]:
    collection = get_collection()
    results = collection.get(include=["metadatas"])

    docs_map: Dict[str, Dict] = {}
    for m in results["metadatas"]:
        doc_id = m.get("doc_id")
        if not doc_id:
            continue
        if doc_id not in docs_map:
            docs_map[doc_id] = {
                "doc_id": doc_id,
                "filename": m.get("filename", m.get("url", "unknown")),
                "chunk_count": 0,
                "created_at": m.get("scraped_at", m.get("created_at", "unknown")),
            }
        docs_map[doc_id]["chunk_count"] += 1

    return list(docs_map.values())


@_chroma_retry
def get_stats() -> Dict[str, int]:
    collection = get_collection()
    return {"total_chunks": collection.count()}
