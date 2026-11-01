"""
Deterministic CHUNK-level retrieval eval (no LLM, no API cost, reproducible).

The source-doc metric is too coarse here (only 3 docs -> always a hit). The real
failure mode is: the right DOCUMENT is retrieved but the specific CHUNK that
answers the question is not surfaced near the top. This measures that directly
using keyword presence in the retrieved chunk text, plus the rank of the first
chunk that actually contains the answer keywords.

Uses the same expected keywords as the original test_queries.json where ids line
up, but keys off gold_qa.json questions with an inline keyword map so it stays
aligned with the gold set actually under test.

Metrics over in-scope questions:
  - keyword_recall@k: avg fraction of a question's answer-keywords present anywhere
                      in the top-k retrieved chunk TEXT.
  - answer_chunk_mrr: mean reciprocal rank of the FIRST retrieved chunk that
                      contains >=50% of the answer keywords (0 if none do).
"""

import json
import asyncio
from typing import List

import eval_common  # noqa: F401
from eval_common import EVAL_OWNER_ID, GOLD_PATH

# Answer-bearing keywords per gold question (lowercased match against chunk text).
KEYWORDS = {
    "g01": ["subset", "artificial intelligence", "data", "learn"],
    "g02": ["layers", "neurons", "weights", "activation"],
    "g03": ["labeled", "unlabeled", "classification", "clustering"],
    "g04": ["pretrained", "fine-tune", "task"],
    "g05": ["optimization", "loss", "learning rate", "steepest"],
    "g06": ["training", "generalize", "validation", "regularization"],
    "g07": ["gradient", "chain rule", "backward", "loss"],
    "g08": ["list", "dictionary", "tuple", "set"],
    "g09": ["decorator", "wrapper", "higher-order", "@"],
    "g10": ["gil", "thread", "cpython", "bytecode"],
    "g11": ["coroutine", "event loop", "await", "non-blocking"],
    "g12": ["reference counting", "cycle", "deallocated"],
    "g13": ["rest", "graphql", "endpoint", "schema"],
    "g14": ["consistency", "availability", "partition"],
    "g15": ["replicas", "converge", "diverge"],
    "g16": ["raft", "leader", "election", "log"],
    "g17": ["microservices", "independent", "deploy"],
    "g18": ["entities", "relationships", "nodes", "edges"],
    "g19": ["container", "image", "kernel", "lightweight"],
    "g20": ["gil", "multiprocessing", "cpu-bound", "parallel"],
    "g21": ["activation function", "weights", "sums"],
    "g22": ["consistency", "availability", "partition"],
}


async def main(top_k: int = 5) -> None:
    from app.rag.retrievers.hybrid_retriever import hybrid_search

    gold = [g for g in json.load(open(GOLD_PATH)) if g["id"] in KEYWORDS]

    recall_sum = 0.0
    rr_sum = 0.0
    rows = []

    for g in gold:
        kws = [k.lower() for k in KEYWORDS[g["id"]]]
        res: List[dict] = await hybrid_search(g["question"], top_k=top_k, owner_id=EVAL_OWNER_ID)
        texts = [(r.get("parent_content") or r.get("text") or "").lower() for r in res]

        combined = " ".join(texts)
        recall = sum(1 for k in kws if k in combined) / len(kws)
        recall_sum += recall

        # rank of first chunk containing >=50% of keywords
        rank = 0
        for i, t in enumerate(texts):
            if sum(1 for k in kws if k in t) / len(kws) >= 0.5:
                rank = i + 1
                break
        rr_sum += (1.0 / rank) if rank else 0.0
        rows.append((g["id"], round(recall, 2), rank))

    n = len(gold)
    print(f"Chunk-level retrieval over {n} in-scope questions (top_k={top_k})\n")
    for qid, rec, rank in rows:
        mark = "MISS" if rank == 0 else ("ok " if rank == 1 else "hit")
        print(f"  [{mark}] {qid}: keyword_recall={rec:.2f} answer_chunk_rank={rank or '-'}")
    print(
        f"\n  keyword_recall@{top_k}: {recall_sum / n:.3f}"
        f"\n  answer_chunk_MRR:    {rr_sum / n:.3f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
