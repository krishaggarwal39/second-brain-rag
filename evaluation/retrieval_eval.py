"""
Deterministic retrieval-quality eval (NO LLM, NO API cost, fully reproducible).

Measures whether hybrid retrieval surfaces the correct source document for each
in-scope gold question. This isolates the retrieval/chunking effect from the
LLM judge (which is subject to free-tier quota limits and scoring noise).

Metrics (over in-scope questions only):
  - source_hit@k:   fraction of questions whose expected source_doc appears in top-k
  - mrr:            mean reciprocal rank of the first chunk from the expected source
  - top1_accuracy:  fraction where the #1 retrieved chunk is from the expected source

Run against whatever is currently ingested in answer_eval_* dirs:
  CHILD_CHUNK_SIZE=250 python retrieval_eval.py   # (re-ingest first if chunking changed)
"""

import os
import json
import asyncio
from typing import List

import eval_common  # noqa: F401  (local-mode env + sys.path)
from eval_common import EVAL_OWNER_ID, GOLD_PATH


async def main(top_k: int = 5) -> None:
    from app.rag.retrievers.hybrid_retriever import hybrid_search

    gold = json.load(open(GOLD_PATH))
    inscope = [g for g in gold if g["category"] != "out_of_scope" and g.get("source_doc")]

    hits = 0
    rr_sum = 0.0
    top1 = 0
    rows = []

    for g in inscope:
        res: List[dict] = await hybrid_search(g["question"], top_k=top_k, owner_id=EVAL_OWNER_ID)
        srcs = [(r.get("filename") or (r.get("metadata") or {}).get("filename", "")) for r in res]
        expected = g["source_doc"]

        rank = next((i + 1 for i, s in enumerate(srcs) if s == expected), 0)
        hit = rank > 0
        hits += int(hit)
        rr_sum += (1.0 / rank) if rank else 0.0
        top1 += int(bool(srcs) and srcs[0] == expected)
        rows.append((g["id"], expected, rank, srcs))

    n = len(inscope)
    print(f"Retrieval eval over {n} in-scope questions (top_k={top_k}, owner='{EVAL_OWNER_ID}')\n")
    for qid, exp, rank, srcs in rows:
        mark = "ok " if rank == 1 else ("hit" if rank else "MISS")
        print(f"  [{mark}] {qid}: expected={exp} rank={rank or '-'}")
    print(
        f"\n  source_hit@{top_k}: {hits}/{n} = {hits / n:.3f}"
        f"\n  MRR:            {rr_sum / n:.3f}"
        f"\n  top1_accuracy:  {top1}/{n} = {top1 / n:.3f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
