"""
Retrieval latency measurement harness (Phase 2, production-readiness step 1).

PURPOSE
  Measure WHERE query latency goes, per sub-stage, so later optimization is
  data-driven. This script ONLY measures — it does not change retrieval behavior,
  models, or thresholds.

WHAT IT MEASURES (per gold question, all local, no Docker)
  * hybrid_search() sub-stages, read from the fine-grained timings the retriever
    now records (see app/rag/retrievers/hybrid_retriever.get_last_hybrid_timings):
      - embedding_ms         query embedding (local SentenceTransformer)
      - dense_ms             ChromaDB vector search
      - sparse_ms            BM25 keyword search (asyncio.to_thread)
      - fusion_hydration_ms  RRF + metadata hydration + owner-isolation filter
      - rerank_ms            CrossEncoder.predict (CPU-bound, asyncio.to_thread)
      - parent_hydration_ms  parent-content hydration from the (in-memory) cache
      - total_ms             sum of the sub-stages above
  * graph_ms                 retrieve_graph_context() timed separately (runs in
                             parallel with hybrid_search in the real pipeline).
  * end_to_end_retrieval_ms  wall-clock of hybrid_search + graph in parallel
                             (mirrors pipeline._retrieve_all_context).
  * generation_ms            OPTIONAL: full pipeline.ask() LLM latency, measured
                             only if GOOGLE_API_KEY is present AND --with-generation
                             is passed and the call actually succeeds. Rate-limited /
                             missing-key runs are SKIPPED and noted, never faked.

COLD vs WARM
  The embedder and CrossEncoder load once (model download/load is a one-time cost,
  not per-query). We warm the models with one throwaway query BEFORE timing, and
  separately record that first cold query's timing so cold-vs-warm is visible.

OUTPUT
  Prints a per-stage summary table and writes evaluation/reports/latency_report.json
  with mean/median/p95/min/max (ms) per stage. NEVER fabricates numbers: stages that
  did not run (e.g. generation when skipped) are recorded as skipped with a reason.

USAGE
  # retrieval only (default; no external API needed):
  GOLD_FILE=gold_qa_well.json python latency_eval.py

  # also measure generation latency (needs a live Gemini/Groq quota):
  GOLD_FILE=gold_qa_well.json python latency_eval.py --with-generation

  # fewer warm iterations / questions for a quick check:
  python latency_eval.py --repeats 1 --limit 5
"""

import os
import sys
import json
import time
import argparse
import asyncio
import statistics
from typing import Dict, List, Optional

import eval_common  # noqa: F401  (sets local-mode env + sys.path before backend imports)
from eval_common import EVAL_OWNER_ID, REPORTS_DIR, GOLD_PATH

REPORT_PATH = os.path.join(REPORTS_DIR, "latency_report.json")

# Sub-stage keys recorded by hybrid_search, in pipeline order.
HYBRID_STAGE_KEYS = [
    "embedding_ms",
    "dense_ms",
    "sparse_ms",
    "fusion_hydration_ms",
    "rerank_ms",
    "parent_hydration_ms",
    "total_ms",
]


def _percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0..100). Returns 0.0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    # nearest-rank: rank = ceil(pct/100 * N), clamped to [1, N]
    import math
    rank = max(1, min(len(ordered), math.ceil((pct / 100.0) * len(ordered))))
    return ordered[rank - 1]


def _summarize(values: List[float]) -> Dict[str, float]:
    """mean/median/p95/min/max over a list of per-query ms values."""
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(_percentile(values, 95), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


async def _time_graph(question: str) -> Optional[float]:
    """Time retrieve_graph_context separately. Returns ms, or None if it errored
    (e.g. Neo4j not running locally / entity-extraction LLM unavailable). We never
    fail the whole run on a graph error — graph is optional context."""
    from app.rag.graph.graph_retriever import retrieve_graph_context
    try:
        t0 = time.perf_counter()
        await retrieve_graph_context(question)
        return (time.perf_counter() - t0) * 1000
    except Exception:
        return None


async def _time_parallel_retrieval(question: str) -> float:
    """Wall-clock of hybrid_search + graph running in parallel, mirroring
    pipeline._retrieve_all_context. Graph errors are swallowed (timing only)."""
    from app.rag.retrievers.hybrid_retriever import hybrid_search
    from app.rag.graph.graph_retriever import retrieve_graph_context

    async def _safe_graph():
        try:
            return await retrieve_graph_context(question)
        except Exception:
            return ""

    t0 = time.perf_counter()
    await asyncio.gather(
        hybrid_search(question, top_k=5, owner_id=EVAL_OWNER_ID),
        _safe_graph(),
    )
    return (time.perf_counter() - t0) * 1000


async def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval latency measurement harness.")
    ap.add_argument("--repeats", type=int, default=3,
                    help="warm timed passes over the gold set (default 3; report aggregates all).")
    ap.add_argument("--limit", type=int, default=None,
                    help="limit number of gold questions (default: all in-scope).")
    ap.add_argument("--with-generation", action="store_true",
                    help="also measure full pipeline.ask() generation latency (needs live LLM quota).")
    ap.add_argument("--graph-sample", type=int, default=5,
                    help="how many questions to time graph retrieval on (default 5). Graph "
                         "entity-extraction calls an external LLM, so we sample rather than "
                         "run it on every question x repeat to avoid burning quota.")
    args = ap.parse_args()

    from app.rag.retrievers.hybrid_retriever import hybrid_search, get_last_hybrid_timings

    gold = json.load(open(GOLD_PATH))
    inscope = [g for g in gold if g["category"] != "out_of_scope" and g.get("source_doc")]
    if args.limit:
        inscope = inscope[: args.limit]
    if not inscope:
        raise SystemExit("ERROR: no in-scope gold questions found; nothing to measure.")

    questions = [g["question"] for g in inscope]
    print(f"Latency eval: {len(questions)} in-scope questions x {args.repeats} warm repeats "
          f"(owner='{EVAL_OWNER_ID}', gold='{os.path.basename(GOLD_PATH)}')\n")

    # --- COLD: first query loads the embedder + CrossEncoder models. Record it
    # separately so the one-time model-load cost is visible but EXCLUDED from the
    # warm per-query aggregates below.
    t_cold = time.perf_counter()
    await hybrid_search(questions[0], top_k=5, owner_id=EVAL_OWNER_ID)
    cold_total_ms = (time.perf_counter() - t_cold) * 1000
    cold_stages = get_last_hybrid_timings() or {}
    print(f"Cold (model-load) first query: {cold_total_ms:.1f} ms wall "
          f"(rerank {cold_stages.get('rerank_ms', 0):.1f} ms, "
          f"embedding {cold_stages.get('embedding_ms', 0):.1f} ms)\n")

    # One extra warm-up pass so caches/thread-pools are hot before timed passes.
    for q in questions:
        await hybrid_search(q, top_k=5, owner_id=EVAL_OWNER_ID)

    # --- WARM timed passes (LOCAL retrieval): collect hybrid per-stage samples
    # across all repeats/questions. This is the precise, external-API-free part.
    hybrid_samples: Dict[str, List[float]] = {k: [] for k in HYBRID_STAGE_KEYS}
    per_run_total_median: List[float] = []

    for r in range(args.repeats):
        run_totals: List[float] = []
        for q in questions:
            await hybrid_search(q, top_k=5, owner_id=EVAL_OWNER_ID)
            t = get_last_hybrid_timings() or {}
            for k in HYBRID_STAGE_KEYS:
                if k in t:
                    hybrid_samples[k].append(t[k])
            run_totals.append(t.get("total_ms", 0.0))

        run_median = statistics.median(run_totals) if run_totals else 0.0
        per_run_total_median.append(round(run_median, 3))
        print(f"  run {r + 1}/{args.repeats}: hybrid total_ms median = {run_median:.2f} ms")

    # --- Graph retrieval (timed SEPARATELY, on a small sample). Graph entity
    # extraction calls an external LLM, so its latency is API/quota-bound and noisy;
    # we sample it rather than run it on every question x repeat. In production it
    # runs in PARALLEL with hybrid_search, so it only adds to retrieval latency when
    # it is slower than hybrid_search (which, being LLM-bound, it usually is).
    graph_samples: List[float] = []
    graph_errors = 0
    parallel_samples: List[float] = []
    graph_sample_qs = questions[: min(args.graph_sample, len(questions))]
    for q in graph_sample_qs:
        g = await _time_graph(q)
        if g is None:
            graph_errors += 1
        else:
            graph_samples.append(g)
        parallel_samples.append(await _time_parallel_retrieval(q))

    # --- Build the report.
    hybrid_summary = {k: _summarize(v) for k, v in hybrid_samples.items()}

    report: Dict = {
        "config": {
            "gold_file": os.path.basename(GOLD_PATH),
            "owner_id": EVAL_OWNER_ID,
            "questions": len(questions),
            "warm_repeats": args.repeats,
            "graph_sample_questions": len(graph_sample_qs),
            "reranker_model": os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "retrieval_multiplier": int(os.getenv("RETRIEVAL_MULTIPLIER", "3")),
            "top_k": 5,
        },
        "cold_first_query": {
            "wall_ms": round(cold_total_ms, 3),
            "stages_ms": {k: round(v, 3) for k, v in cold_stages.items()},
            "note": "One-time model load (embedder + CrossEncoder) is included here and "
                    "EXCLUDED from the warm aggregates below.",
        },
        "warm_hybrid_stages_ms": hybrid_summary,
        "warm_hybrid_total_ms_median_per_run": per_run_total_median,
        "graph_retrieval_ms": _summarize(graph_samples),
        "end_to_end_parallel_retrieval_ms": _summarize(parallel_samples),
        "generation_ms": None,  # filled below or marked skipped
    }

    if graph_errors:
        report["graph_retrieval_note"] = (
            f"{graph_errors} graph calls returned no timing (Neo4j not running locally "
            f"and/or entity-extraction LLM unavailable). Graph runs in PARALLEL with "
            f"hybrid_search in production, so it does not add to retrieval latency unless "
            f"it is slower than hybrid_search."
        )

    # --- Dominant sub-stage (data-driven, by median of warm samples). total_ms is
    # excluded since it is the sum, not a stage.
    stage_medians = {
        k: hybrid_summary[k]["median"]
        for k in HYBRID_STAGE_KEYS
        if k != "total_ms" and hybrid_summary[k]["n"] > 0
    }
    if stage_medians:
        dominant = max(stage_medians, key=stage_medians.get)
        total_median = hybrid_summary["total_ms"]["median"] or 1.0
        report["dominant_substage"] = {
            "stage": dominant,
            "median_ms": stage_medians[dominant],
            "share_of_total_pct": round(100.0 * stage_medians[dominant] / total_median, 1),
        }

    # --- Optional generation latency (external LLM; never fabricate, never hard-fail).
    gen_block: Dict
    if not args.with_generation:
        gen_block = {"measured": False, "reason": "not requested (pass --with-generation to measure)."}
    elif not os.getenv("GOOGLE_API_KEY") and not os.getenv("GROQ_API_KEY"):
        gen_block = {"measured": False,
                     "reason": "no GOOGLE_API_KEY / GROQ_API_KEY available; generation skipped."}
    else:
        gen_block = await _measure_generation(questions)

    report["generation_ms"] = gen_block

    # --- Write + print.
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    _print_summary(report)
    print(f"\nWrote {REPORT_PATH}")


async def _measure_generation(questions: List[str]) -> Dict:
    """Measure full pipeline.ask() latency on a small sample. If the LLM is
    rate-limited / errors, skip those samples and note why — never fabricate."""
    from app.rag.pipeline import pipeline, is_error_placeholder

    sample = questions[: min(5, len(questions))]
    gen_samples: List[float] = []
    total_samples: List[float] = []
    errors = 0
    last_err = ""
    for q in sample:
        try:
            res = await pipeline.ask(q, owner_id=EVAL_OWNER_ID)
            answer = res.get("answer", "")
            if is_error_placeholder(answer):
                errors += 1
                last_err = (answer or "empty/error placeholder")[:120]
                continue
            timings = res.get("timings", {})
            gen_samples.append(float(timings.get("generation_ms", 0.0)))
            total_samples.append(float(timings.get("total_ms", 0.0)))
        except Exception as e:
            errors += 1
            last_err = f"{e.__class__.__name__}: {e}"[:120]

    if not gen_samples:
        return {"measured": False,
                "reason": f"all {len(sample)} generation calls failed/rate-limited "
                          f"(last: {last_err or 'unknown'}). Retrieval numbers above are unaffected."}

    return {
        "measured": True,
        "sample_size": len(gen_samples),
        "attempted": len(sample),
        "errors": errors,
        "generation_ms": _summarize(gen_samples),
        "total_ms": _summarize(total_samples),
        "note": "LLM latency depends on external Gemini/Groq APIs and quotas; it is "
                "variable run-to-run and separate from the local retrieval numbers.",
    }


def _print_summary(report: Dict) -> None:
    print("\n=== RETRIEVAL LATENCY (warm, per-stage, ms) ===")
    hdr = f"{'stage':<22}{'n':>5}{'mean':>10}{'median':>10}{'p95':>10}{'min':>9}{'max':>10}"
    print(hdr)
    print("-" * len(hdr))
    for k in HYBRID_STAGE_KEYS:
        s = report["warm_hybrid_stages_ms"][k]
        label = k if k != "total_ms" else "TOTAL (sum of above)"
        print(f"{label:<22}{s['n']:>5}{s['mean']:>10.2f}{s['median']:>10.2f}"
              f"{s['p95']:>10.2f}{s['min']:>9.2f}{s['max']:>10.2f}")

    g = report["graph_retrieval_ms"]
    print(f"\n{'graph (parallel)':<22}{g['n']:>5}{g['mean']:>10.2f}{g['median']:>10.2f}"
          f"{g['p95']:>10.2f}{g['min']:>9.2f}{g['max']:>10.2f}")
    e = report["end_to_end_parallel_retrieval_ms"]
    print(f"{'end_to_end_parallel':<22}{e['n']:>5}{e['mean']:>10.2f}{e['median']:>10.2f}"
          f"{e['p95']:>10.2f}{e['min']:>9.2f}{e['max']:>10.2f}")

    if "dominant_substage" in report:
        d = report["dominant_substage"]
        print(f"\nDominant sub-stage: {d['stage']} "
              f"(median {d['median_ms']:.2f} ms = {d['share_of_total_pct']}% of hybrid total)")

    c = report["cold_first_query"]
    print(f"\nCold first query (model load incl.): {c['wall_ms']:.1f} ms "
          f"(excluded from warm aggregates)")

    gen = report.get("generation_ms") or {}
    if gen.get("measured"):
        gm = gen["generation_ms"]
        print(f"\nGeneration (LLM, n={gen['sample_size']}): median {gm['median']:.1f} ms, "
              f"p95 {gm['p95']:.1f} ms  [external API]")
    else:
        print(f"\nGeneration: SKIPPED — {gen.get('reason', 'n/a')}")


if __name__ == "__main__":
    asyncio.run(main())
