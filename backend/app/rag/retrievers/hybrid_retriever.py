import os
import re
import time
import asyncio
import uuid
import contextvars
from typing import List, Dict, Optional

import structlog
from sentence_transformers import CrossEncoder

from app.rag.vectorstore.chroma_store import query as chroma_query, get_chunk_metadatas
from app.rag.embeddings.local_embedder import embed_documents
from app.rag.vectorstore.bm25_store import get_bm25_store
from app.core.cache import get_cache, set_cache, increment_metric
from app.core.telemetry import get_tracer

logger = structlog.get_logger(__name__)

# Per-call fine-grained retrieval timings (milliseconds). hybrid_search() sets this
# on every call so callers (e.g. the latency eval harness) can read the sub-stage
# breakdown of the most recent invocation in the current async context without
# changing hybrid_search's return type. Timing-only; does not affect results.
_last_hybrid_timings: contextvars.ContextVar[Optional[Dict[str, float]]] = contextvars.ContextVar(
    "last_hybrid_timings", default=None
)


def get_last_hybrid_timings() -> Optional[Dict[str, float]]:
    """Return the per-stage timing dict (ms) from the most recent hybrid_search
    call in the current async context, or None if none has run."""
    return _last_hybrid_timings.get()

# --- Configuration ---
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RRF_K = int(os.getenv("RRF_K", "60"))
RETRIEVAL_MULTIPLIER = int(os.getenv("RETRIEVAL_MULTIPLIER", "3"))
# USE_MMR / MMR_LAMBDA are read per call in hybrid_search (no module-level copies).

# Load local cross-encoder for reranking (lazy-safe: if model download fails, reranking is skipped).
try:
    reranker = CrossEncoder(RERANKER_MODEL)
    logger.info("CrossEncoder loaded for reranking.", model=RERANKER_MODEL)
except Exception as e:
    logger.error(f"Failed to load CrossEncoder: {e}")
    reranker = None


def reciprocal_rank_fusion(
    dense_results: List[Dict], sparse_results: List[Dict], k: int = RRF_K
) -> List[Dict]:
    """Merge dense and sparse results using Reciprocal Rank Fusion."""
    fused_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    def add_to_fusion(results: List[Dict]):
        for rank, doc in enumerate(results):
            doc_id = (
                doc.get("id")
                or doc.get("metadata", {}).get("hash")
                or str(uuid.uuid4())
            )
            doc["id"] = doc_id

            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 1.0 / (rank + k)

    add_to_fusion(dense_results)
    add_to_fusion(sparse_results)

    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in sorted_docs]


def _copy_metadata_to_doc(doc: Dict) -> None:
    """Consistently copy metadata fields (filename, page_number) to top-level doc."""
    meta = doc.get("metadata") or {}
    if not doc.get("filename"):
        fname = meta.get("filename") or meta.get("file_name")
        if fname:
            doc["filename"] = fname
    if doc.get("page_number") is None:
        page_num = meta.get("page_number") if meta.get("page_number") is not None else meta.get("page")
        if page_num is not None:
            doc["page_number"] = page_num


async def _hydrate_parent_content(doc: Dict) -> None:
    """Hydrate parent chunk content with read-through fallback to sparse store DB (R-12)."""
    meta = doc.get("metadata", {})
    parent_id = meta.get("parent_id")
    if parent_id:
        parent_text = await get_cache(f"parent:{parent_id}")
        if not parent_text:
            # R-12: Read-through fallback to durable sparse store
            parent_text = await asyncio.to_thread(get_bm25_store().get_parent, parent_id)
            if parent_text:
                await set_cache(f"parent:{parent_id}", parent_text, ttl=None)
        if parent_text:
            doc["parent_content"] = parent_text
        else:
            logger.warning(
                "parent_missing",
                parent_id=parent_id,
                doc_id=meta.get("doc_id") or doc.get("doc_id"),
            )
            await increment_metric("parent_missing", 1)


def _token_jaccard_similarity(tokens_a: set, tokens_b: set) -> float:
    """Calculate token Jaccard similarity between two sets of tokens."""
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def _mmr_select(
    candidates: List[Dict],
    top_k: int,
    mmr_lambda: float = 0.7,
) -> List[Dict]:
    """Select top_k diverse candidates using Maximal Marginal Relevance (MMR).

    Greedy MMR:
      1. Start from the highest-relevance candidate.
      2. Iteratively pick candidate maximizing:
           score = lambda * norm_relevance(c) - (1 - lambda) * max(similarity(c, s) for s in already_selected)
      3. Stop at top_k candidates.
    Lexical similarity is computed via token Jaccard similarity on lowercased word sets.
    """
    if not candidates or top_k <= 0:
        return []
    if len(candidates) <= 1 or top_k == 1:
        return candidates[:top_k]

    token_sets = [
        set(re.findall(r"\w+", (doc.get("text") or "").lower()))
        for doc in candidates
    ]
    scores = [float(doc.get("rerank_score", 0.0)) for doc in candidates]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    if score_range > 1e-9:
        norm_scores = [(s - min_score) / score_range for s in scores]
    else:
        norm_scores = [1.0 for _ in scores]

    selected_indices: List[int] = [0]
    remaining_indices = list(range(1, len(candidates)))
    target_count = min(top_k, len(candidates))

    while len(selected_indices) < target_count and remaining_indices:
        best_idx = None
        best_mmr_score = -float("inf")

        for idx in remaining_indices:
            cand_tokens = token_sets[idx]
            max_sim = max(
                _token_jaccard_similarity(cand_tokens, token_sets[s_idx])
                for s_idx in selected_indices
            )
            mmr_score = mmr_lambda * norm_scores[idx] - (1.0 - mmr_lambda) * max_sim
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx is None:
            best_idx = remaining_indices[0]

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]


async def hybrid_search(
    query: str, top_k: int = 5, owner_id: Optional[str] = None
) -> List[Dict]:
    """
    Full hybrid retrieval pipeline:
    1. Dense search (ChromaDB embeddings with optional owner_id filter)
    2. Sparse search (BM25 keyword matching)
    3. Reciprocal Rank Fusion
    4. Metadata hydration & per-user document isolation filter (via Chroma metadata)
    5. Cross-Encoder reranking
    6. Parent content and metadata hydration from cache and Chroma
    """
    fetch_k = top_k * RETRIEVAL_MULTIPLIER

    # Fine-grained per-stage timings (ms). Recorded with time.perf_counter() around
    # each stage, attached to the active OTEL span, and exposed via the module-level
    # contextvar. Timing-only instrumentation — it does not change results.
    timings: Dict[str, float] = {}
    tracer = get_tracer("rag.hybrid_retriever")

    with tracer.start_as_current_span("hybrid_search") as span:

        def _finish(results: List[Dict]) -> List[Dict]:
            """Attach timings to the span + contextvar, then return results unchanged."""
            timings["total_ms"] = sum(
                v for k, v in timings.items() if k.endswith("_ms") and k != "total_ms"
            )
            for key, val in timings.items():
                span.set_attribute(f"hybrid.{key}", val)
            _last_hybrid_timings.set(dict(timings))
            return results

        # 1. Dense search: query embedding + ChromaDB vector search.
        dense_results: List[Dict] = []
        _t = time.perf_counter()
        embeddings = await embed_documents([query])
        timings["embedding_ms"] = (time.perf_counter() - _t) * 1000

        _t = time.perf_counter()
        if embeddings:
            filters = {"owner_id": str(owner_id)} if owner_id else None
            dense_results = await chroma_query(
                embeddings[0], n_results=fetch_k, score_threshold=0.0, filters=filters
            )
        timings["dense_ms"] = (time.perf_counter() - _t) * 1000

        # 2. Sparse search (BM25) - owner-filtered at store level
        _t = time.perf_counter()
        bm25_store = get_bm25_store()
        sparse_results = await asyncio.to_thread(
            bm25_store.search, query, fetch_k, owner_id=owner_id
        )
        timings["sparse_ms"] = (time.perf_counter() - _t) * 1000

        # 3. RRF Fusion + metadata hydration + owner-isolation filter
        _t = time.perf_counter()
        fused_results = reciprocal_rank_fusion(dense_results, sparse_results)

        # Hydrate metadata from Chroma for chunks missing metadata (e.g. from BM25 sparse search)
        missing_ids = [doc["id"] for doc in fused_results if doc.get("id") and not doc.get("metadata")]
        if missing_ids:
            meta_map = await get_chunk_metadatas(missing_ids)
            for doc in fused_results:
                if not doc.get("metadata") and doc.get("id") in meta_map:
                    doc["metadata"] = meta_map[doc["id"]]

        for doc in fused_results:
            _copy_metadata_to_doc(doc)

        # User isolation: restrict fused results to owner_id if specified (defense-in-depth).
        if owner_id:
            target_owner = str(owner_id)
            fused_results = [
                doc for doc in fused_results
                if doc.get("metadata", {}).get("owner_id") == target_owner
            ]
        timings["fusion_hydration_ms"] = (time.perf_counter() - _t) * 1000

        if not reranker or not fused_results:
            timings["rerank_ms"] = 0.0
            _t = time.perf_counter()
            for doc in fused_results:
                await _hydrate_parent_content(doc)
                _copy_metadata_to_doc(doc)
            timings["parent_hydration_ms"] = (time.perf_counter() - _t) * 1000
            timings["mmr_ms"] = 0.0
            return _finish(fused_results[:top_k])

        # 4. Reranking (Cross-Encoder) — CPU-bound, runs in thread pool
        pairs = [[query, doc.get("text", "")] for doc in fused_results]

        try:
            _t = time.perf_counter()
            scores = await asyncio.to_thread(reranker.predict, pairs)
            timings["rerank_ms"] = (time.perf_counter() - _t) * 1000

            # 5. Hydrate parent content from cache for prompt building
            _t = time.perf_counter()
            for doc, score in zip(fused_results, scores):
                doc["rerank_score"] = float(score)
                await _hydrate_parent_content(doc)
                _copy_metadata_to_doc(doc)
            timings["parent_hydration_ms"] = (time.perf_counter() - _t) * 1000

            reranked_results = sorted(
                fused_results, key=lambda x: x["rerank_score"], reverse=True
            )

            # 6. MMR (Maximal Marginal Relevance) diversity selection
            use_mmr = os.getenv("USE_MMR", "1").strip().lower() not in ("0", "false", "no")
            try:
                mmr_lambda = float(os.getenv("MMR_LAMBDA", "0.7"))
            except ValueError:
                mmr_lambda = 0.7

            if not use_mmr:
                timings["mmr_ms"] = 0.0
                return _finish(reranked_results[:top_k])

            _t = time.perf_counter()
            try:
                selected_results = _mmr_select(
                    candidates=reranked_results,
                    top_k=top_k,
                    mmr_lambda=mmr_lambda,
                )
                timings["mmr_ms"] = (time.perf_counter() - _t) * 1000
                return _finish(selected_results)
            except Exception as e:
                logger.error(f"MMR selection failed, falling back to sorted top_k: {e}")
                timings["mmr_ms"] = (time.perf_counter() - _t) * 1000
                return _finish(reranked_results[:top_k])
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            timings.setdefault("rerank_ms", (time.perf_counter() - _t) * 1000)
            timings.setdefault("parent_hydration_ms", 0.0)
            timings.setdefault("mmr_ms", 0.0)
            for doc in fused_results:
                _copy_metadata_to_doc(doc)
            return _finish(fused_results[:top_k])
