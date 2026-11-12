"""
Custom RAG Evaluation Pipeline

Measures retrieval quality WITHOUT requiring paid LLM APIs:
1. Context Recall — Do retrieved chunks contain the expected keywords?
2. Context Precision — Are the top-k results from the expected source?
3. Faithfulness (mock) — Does the system refuse out-of-scope questions?
4. Retrieval Latency — How fast is the hybrid retrieval?

This is a keyword-overlap retrieval evaluation (not the ragas library).
Results are written to evaluation/reports/eval_report.json
"""

import os
import sys
import json
import time
import asyncio
from typing import Dict, List

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Force local mode
os.environ.setdefault("CACHE_BACKEND", "memory")
os.environ.setdefault("CHROMA_PERSIST_DIR", os.path.join(os.path.dirname(__file__), "eval_chroma_data"))
os.environ.setdefault("BM25_DATA_DIR", os.path.join(os.path.dirname(__file__), "eval_bm25_data"))
os.environ.setdefault("OTEL_ENABLED", "false")
EVAL_OWNER_ID = os.getenv("EVAL_OWNER_ID", "eval-owner")
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "-5.0"))

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Thresholds (configurable via env)
MIN_CONTEXT_RECALL = float(os.getenv("RAGAS_MIN_RECALL", "0.70"))
MIN_CONTEXT_PRECISION = float(os.getenv("RAGAS_MIN_PRECISION", "0.60"))


def load_test_queries() -> List[Dict]:
    queries_path = os.path.join(os.path.dirname(__file__), "test_queries.json")
    with open(queries_path) as f:
        return json.load(f)


def compute_keyword_recall(retrieved_texts: List[str], expected_keywords: List[str]) -> float:
    """What fraction of expected keywords appear in the retrieved context?"""
    if not expected_keywords:
        raise ValueError("in-scope queries need expected_keywords")

    combined_text = " ".join(retrieved_texts).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    return hits / len(expected_keywords)


def compute_source_precision(retrieved_metadata: List[Dict], expected_source: str) -> float:
    """What fraction of top-k results come from the expected source document?"""
    if not expected_source:
        raise ValueError("in-scope queries need expected_source")
    if not retrieved_metadata:
        return 0.0  # nothing retrieved is a precision failure, not a pass

    correct = sum(
        1 for m in retrieved_metadata if m.get("filename", "") == expected_source
    )
    return correct / len(retrieved_metadata)


def out_of_scope_passes(retrieved: List[Dict], threshold: float = RERANK_THRESHOLD) -> bool:
    """An off-topic query passes only if relevance pruning leaves NO context.

    That is the condition under which the pipeline refuses without calling the LLM.
    Previously out-of-scope queries were auto-passed with recall 1.00.
    """
    from app.rag.context_engineering import prune_irrelevant_context

    return len(prune_irrelevant_context(retrieved, threshold=threshold)) == 0


async def evaluate():
    from app.rag.retrievers.hybrid_retriever import hybrid_search

    queries = load_test_queries()
    print(f"Running evaluation on {len(queries)} test queries...\n")

    results = []
    total_recall = 0.0
    total_precision = 0.0
    total_latency = 0.0
    scorable_count = 0

    for q in queries:
        qid = q["id"]
        question = q["question"]
        expected_kw = q["expected_keywords"]
        expected_src = q["expected_source"]
        category = q["category"]

        t0 = time.time()
        try:
            retrieved = await hybrid_search(question, top_k=5, owner_id=EVAL_OWNER_ID)
        except Exception as e:
            print(f"  [{qid}] ERROR: {e}")
            results.append({
                "id": qid,
                "question": question,
                "category": category,
                "error": str(e),
            })
            continue
        latency_ms = (time.time() - t0) * 1000

        texts = [r.get("text", "") for r in retrieved]
        metadatas = [r.get("metadata", {}) for r in retrieved]

        if category == "out_of_scope":
            # Recall/precision are not defined for off-topic questions.
            recall = precision = None
            status = "PASS" if out_of_scope_passes(retrieved) else "FAIL"
        else:
            recall = compute_keyword_recall(texts, expected_kw)
            precision = compute_source_precision(metadatas, expected_src)
            status = "PASS" if recall >= MIN_CONTEXT_RECALL else "FAIL"

        result = {
            "id": qid,
            "question": question,
            "category": category,
            "context_recall": None if recall is None else round(recall, 3),
            "source_precision": None if precision is None else round(precision, 3),
            "latency_ms": round(latency_ms, 1),
            "retrieved_count": len(retrieved),
            "status": status,
        }
        results.append(result)

        if category != "out_of_scope":
            total_recall += recall
            total_precision += precision
            scorable_count += 1

        total_latency += latency_ms

        icon = "✓" if status == "PASS" else "✗"
        if recall is None:
            print(f"  [{icon}] {qid}: out-of-scope, context after pruning={'none' if status == 'PASS' else 'present'} ({latency_ms:.0f}ms) - {question[:50]}")
        else:
            print(f"  [{icon}] {qid}: recall={recall:.2f} precision={precision:.2f} ({latency_ms:.0f}ms) - {question[:50]}")

    # Compute aggregates
    avg_recall = total_recall / max(scorable_count, 1)
    avg_precision = total_precision / max(scorable_count, 1)
    avg_latency = total_latency / len(queries)
    pass_count = sum(1 for r in results if r.get("status") == "PASS")

    report = {
        "summary": {
            "total_queries": len(queries),
            "passed": pass_count,
            "failed": len(queries) - pass_count,
            "avg_context_recall": round(avg_recall, 3),
            "avg_source_precision": round(avg_precision, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "thresholds": {
                "min_context_recall": MIN_CONTEXT_RECALL,
                "min_context_precision": MIN_CONTEXT_PRECISION,
            },
        },
        "results": results,
    }

    # Write report
    report_path = os.path.join(REPORTS_DIR, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Queries:           {len(queries)}")
    print(f"  Passed:            {pass_count}/{len(queries)}")
    print(f"  Avg Context Recall:   {avg_recall:.3f} (threshold: {MIN_CONTEXT_RECALL})")
    print(f"  Avg Source Precision: {avg_precision:.3f} (threshold: {MIN_CONTEXT_PRECISION})")
    print(f"  Avg Latency:          {avg_latency:.1f}ms")
    print(f"\n  Report saved to: {report_path}")

    # Exit with error if thresholds are not met
    if avg_recall < MIN_CONTEXT_RECALL:
        print(f"\n  FAIL: Context recall {avg_recall:.3f} < threshold {MIN_CONTEXT_RECALL}")
        sys.exit(1)
    if avg_precision < MIN_CONTEXT_PRECISION:
        print(f"\n  FAIL: Source precision {avg_precision:.3f} < threshold {MIN_CONTEXT_PRECISION}")
        sys.exit(1)

    oos_failed = [r["id"] for r in results if r.get("category") == "out_of_scope" and r.get("status") == "FAIL"]
    if oos_failed and os.getenv("EVAL_REQUIRE_OOS", "true").lower() == "true":
        print(f"\n  FAIL: off-topic queries still retrieved relevant-looking context: {', '.join(oos_failed)}")
        sys.exit(1)

    print(f"\n  ALL THRESHOLDS MET ✓")


if __name__ == "__main__":
    asyncio.run(evaluate())
