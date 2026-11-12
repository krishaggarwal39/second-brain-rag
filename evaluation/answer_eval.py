"""
Answer-quality evaluation harness (LLM-as-judge).

For each gold question:
  1. Generate an answer through the REAL owner-scoped pipeline:
     pipeline.ask(question, owner_id=EVAL_OWNER_ID)  -- same call chat.py makes.
  2. Judge the answer with Groq (LLaMa 3.3 70B) against the gold reference and
     the retrieved context, scoring three dimensions 0.0-1.0:
       - faithfulness:       is the answer grounded in the retrieved context (no invented facts)?
       - answer_correctness: does the answer match the reference answer?
       - context_relevance:  did retrieval surface the right source material?

Generator = Gemini (pipeline default). Judge = Groq (different model family,
so the judge is not grading its own output).

Determinism / robustness (zero-bug design):
  - judge temperature = 0
  - judge output is strict JSON; parse failures retried up to 3x
  - 429 (rate limit) retried with exponential backoff
  - out-of-scope questions: a proper refusal scores high on all three dims

Usage:
  python answer_eval.py            # full gold set
  python answer_eval.py --limit 3  # smoke test on first 3 questions
  python answer_eval.py --compare-prev   # diff aggregates vs the previous report

Writes evaluation/reports/answer_quality_report.json and prints a summary table.
NEVER fabricates scores: if generation or judging fails, that question is recorded
as an error and excluded from aggregates.
"""

import os
import sys
import json
import time
import asyncio
import argparse
from typing import Dict, List, Optional

import httpx

import eval_common  # noqa: F401  (sets local-mode env + sys.path before backend imports)
from eval_common import EVAL_OWNER_ID, REPORTS_DIR, GOLD_PATH, require_api_keys

REPORT_PATH = os.path.join(REPORTS_DIR, "answer_quality_report.json")

# --- Judge provider selection ---------------------------------------------
# The judge runs on a DIFFERENT provider/model family than the app's generator
# (Gemini), so it never grades its own output.
# Provider chosen via JUDGE_PROVIDER env ("openrouter" | "groq"); if unset,
# prefer OpenRouter when its key is present, else Groq.
#   - openrouter: its own quota, decoupled from the app's Gemini/Groq budget.
#   - groq: gpt-oss-120b (the old llama-3.3-70b-versatile default was decommissioned).
_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
_JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "openrouter" if _OPENROUTER_KEY else "groq").lower()
if _JUDGE_PROVIDER == "openrouter":
    JUDGE_PROVIDER = "openrouter"
    JUDGE_URL = "https://openrouter.ai/api/v1/chat/completions"
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "meta-llama/llama-3.3-70b-instruct")
    _JUDGE_KEY = _OPENROUTER_KEY
else:
    JUDGE_PROVIDER = "groq"
    JUDGE_URL = "https://api.groq.com/openai/v1/chat/completions"
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
    _JUDGE_KEY = os.getenv("GROQ_API_KEY", "")

JUDGE_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "2"))
JUDGE_MAX_BACKOFF = 8  # cap per-retry wait (s). Kept small on purpose: if the free-tier
# quota is exhausted, we do NOT grind against the limit — we skip the question (its
# score stays absent) and the judge cache preserves everything already scored. Resume
# the full run when quota resets; cached questions are not re-charged.
# Persistent per-(question, answer, judge model) score cache so a run completes even
# when rate limits force many retries, and re-runs only re-judge changed answers.
JUDGE_CACHE_PATH = os.path.join(REPORTS_DIR, ".judge_cache.json")
# Config tag MUST change whenever the pipeline config under test changes (chunking,
# retrieval, prompt). Answers are cached per (CONFIG_TAG, question) so re-runs of the
# same config never re-generate (avoids generator 429s), but a new config regenerates.
CONFIG_TAG = os.getenv("EVAL_CONFIG_TAG", "baseline")
ANSWER_CACHE_PATH = os.path.join(REPORTS_DIR, ".answer_cache.json")
# Delay between questions to stay under free-tier per-minute rate limits.
# With the judge on OpenRouter (separate quota), the only shared constraint is
# the app generator's own limits, so modest pacing suffices. Tunable.
PACING_SECONDS = float(os.getenv("EVAL_PACING_SECONDS", "4.0"))
REFUSAL_MARKER = "cannot answer this based on the provided documents"

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator of a Retrieval-Augmented Generation (RAG) "
    "system's answers. You are given a QUESTION, the GENERATED ANSWER, a REFERENCE "
    "ANSWER (ground truth), and the RETRIEVED CONTEXT the system used. Score the "
    "generated answer on three dimensions, each a float from 0.0 to 1.0:\n"
    "  - faithfulness: 1.0 if every claim in the answer is supported by the retrieved "
    "context; lower if it invents facts not in the context.\n"
    "  - answer_correctness: 1.0 if the answer conveys the same key facts as the "
    "reference answer; lower for missing or wrong facts.\n"
    "  - context_relevance: 1.0 if the retrieved context clearly contains the "
    "information needed to answer; lower if it is off-topic or missing.\n"
    "SPECIAL CASE — if the QUESTION is out of scope (its reference answer says it "
    "cannot be answered from the documents): a generated answer that correctly refuses "
    "or says it cannot answer from the documents should score 1.0 on ALL three "
    "dimensions; an answer that fabricates a response should score low (<= 0.3).\n"
    "Respond with ONLY a JSON object, no prose, no markdown fences, exactly:\n"
    '{"faithfulness": <float>, "answer_correctness": <float>, '
    '"context_relevance": <float>, "reason": "<one short sentence>"}'
)


def load_gold() -> List[Dict]:
    with open(GOLD_PATH) as f:
        return json.load(f)


def _format_context(citations: List[Dict]) -> str:
    """Build a readable context string from the pipeline's citations."""
    if not citations:
        return "(no context retrieved)"
    parts = []
    for i, c in enumerate(citations, 1):
        src = c.get("filename", "unknown")
        excerpt = (c.get("excerpt") or "").strip()
        parts.append(f"[{i}] (source: {src}) {excerpt}")
    return "\n".join(parts)


def _coerce_score(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # Clamp into [0, 1] rather than trusting the model blindly.
    return max(0.0, min(1.0, f))


def _load_judge_cache() -> Dict:
    try:
        with open(JUDGE_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_judge_cache(cache: Dict) -> None:
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        with open(JUDGE_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def _judge_cache_key(question: str, answer: str) -> str:
    import hashlib
    return hashlib.sha256(f"{JUDGE_MODEL}\x00{question}\x00{answer}".encode()).hexdigest()


def _load_answer_cache() -> Dict:
    try:
        with open(ANSWER_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_answer_cache(cache: Dict) -> None:
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        with open(ANSWER_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


async def judge_answer(
    question: str,
    generated_answer: str,
    reference_answer: str,
    context_str: str,
    is_out_of_scope: bool,
) -> Optional[Dict]:
    """Call the judge LLM (OpenRouter or Groq). Returns parsed scores dict, or None on failure."""
    api_key = _JUDGE_KEY
    user_content = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference_answer}\n\n"
        f"RETRIEVED CONTEXT:\n{context_str}\n\n"
        f"GENERATED ANSWER:\n{generated_answer}\n\n"
        f"(This question is {'OUT OF SCOPE' if is_out_of_scope else 'in scope'}.)\n"
        "Return the JSON object now."
    )
    payload = {
        "model": JUDGE_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(JUDGE_URL, headers=headers, json=payload, timeout=60.0)
            if resp.status_code == 429:
                wait = min(2 ** attempt, JUDGE_MAX_BACKOFF)
                print(f"    judge 429 (attempt {attempt + 1}/{JUDGE_MAX_RETRIES}) — waiting {wait}s")
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            scores = {
                "faithfulness": _coerce_score(parsed.get("faithfulness")),
                "answer_correctness": _coerce_score(parsed.get("answer_correctness")),
                "context_relevance": _coerce_score(parsed.get("context_relevance")),
                "reason": str(parsed.get("reason", ""))[:300],
            }
            if any(scores[k] is None for k in ("faithfulness", "answer_correctness", "context_relevance")):
                raise ValueError(f"non-numeric score in judge output: {parsed}")
            return scores
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"    judge parse error (attempt {attempt + 1}): {e}")
            await asyncio.sleep(1)
            continue
        except httpx.HTTPStatusError as e:
            print(f"    judge HTTP error {e.response.status_code} (attempt {attempt + 1})")
            await asyncio.sleep(2 ** attempt)
            continue
        except Exception as e:
            print(f"    judge call failed (attempt {attempt + 1}): {e.__class__.__name__}: {e}")
            await asyncio.sleep(1)
            continue
    return None


async def _dry_answer(item: Dict) -> Dict:
    """Deterministic stand-in for pipeline.ask (no LLM, no network)."""
    if item["category"] == "out_of_scope":
        return {"answer": "I cannot answer this based on the provided documents.", "citations": []}
    return {"answer": item["reference_answer"], "citations": [{"filename": "dry-run", "excerpt": ""}]}


async def run(limit: Optional[int] = None, dry_run: bool = False) -> Dict:
    """Score answers. --dry-run exercises the whole harness with fakes (CI smoke test):
    no API keys, no LLM calls, no cache writes, separate report file."""
    global REPORT_PATH
    if dry_run:
        REPORT_PATH = REPORT_PATH.replace(".json", "_dryrun.json")
        pipeline = None
    else:
        require_api_keys()
        from app.rag.pipeline import pipeline

    gold = load_gold()
    if limit:
        gold = gold[:limit]
    print(f"Evaluating {len(gold)} gold questions (owner_id='{EVAL_OWNER_ID}')...\n")

    judge_cache = _load_judge_cache()
    answer_cache = _load_answer_cache()
    print(f"Config tag: '{CONFIG_TAG}'  (answers cached per config; judge cached per model)")
    results = []
    for idx, item in enumerate(gold):
        qid = item["id"]
        question = item["question"]
        reference = item["reference_answer"]
        category = item["category"]
        is_oos = category == "out_of_scope"

        # Answer cache, keyed by (config tag, question). Lets a run reuse a previously
        # GENERATED answer so generator 429s don't force re-generation of good answers.
        # EVAL_CONFIG_TAG MUST change whenever the pipeline config under test changes
        # (chunking, retrieval, prompt) so answers are regenerated for the new config.
        ans_key = f"{CONFIG_TAG}\x00{qid}\x00{question}"
        cached_ans = None if dry_run else answer_cache.get(ans_key)
        t0 = time.time()
        if cached_ans is not None:
            answer_obj = cached_ans
            gen_ms = 0.0
        else:
            try:
                answer_obj = (await _dry_answer(item)) if dry_run else await pipeline.ask(question, owner_id=EVAL_OWNER_ID)
            except Exception as e:
                print(f"  [{qid}] GENERATION ERROR: {e.__class__.__name__}: {e}")
                results.append({"id": qid, "question": question, "category": category,
                                "error": f"generation: {e.__class__.__name__}: {e}"})
                continue
            gen_ms = (time.time() - t0) * 1000

        answer = (answer_obj.get("answer") or "").strip()
        citations = answer_obj.get("citations", [])
        context_str = _format_context(citations)

        if not answer:
            print(f"  [{qid}] empty answer from pipeline")
            results.append({"id": qid, "question": question, "category": category,
                            "error": "generation: empty answer"})
            continue

        # Exclude provider/infrastructure failures (e.g. the generator hitting a 429 and
        # returning an error placeholder). These are NOT answer-quality signals and must
        # not be judged/averaged, or they silently poison the aggregate.
        _err_markers = ("API Error (", "Check server logs.", "Rate Limit Exceeded",
                        "Streaming error occurred", "No response from")
        if any(m in answer for m in _err_markers):
            print(f"  [{qid}] SKIPPED: provider error placeholder (not a quality signal)")
            results.append({"id": qid, "question": question, "category": category,
                            "answer": answer, "error": "generation: provider error placeholder"})
            continue

        # Persist this good answer so re-runs of the SAME config never re-generate it.
        if cached_ans is None and not dry_run:
            answer_cache[ans_key] = {"answer": answer_obj.get("answer"),
                                     "citations": citations}
            _save_answer_cache(answer_cache)

        cache_key = _judge_cache_key(question, answer)
        scores = None if dry_run else judge_cache.get(cache_key)
        if dry_run:
            scores = {"faithfulness": 1.0, "answer_correctness": 1.0, "context_relevance": 1.0, "reason": "dry-run"}
        elif scores is None:
            # Only spend quota (and pace) on a cache miss.
            if PACING_SECONDS > 0:
                await asyncio.sleep(PACING_SECONDS)
            scores = await judge_answer(question, answer, reference, context_str, is_oos)
            if scores is not None:
                judge_cache[cache_key] = scores
                _save_judge_cache(judge_cache)
        else:
            print(f"    [{qid}] judge: cache hit")

        if scores is None:
            print(f"  [{qid}] JUDGE FAILED after {JUDGE_MAX_RETRIES} attempts")
            results.append({"id": qid, "question": question, "category": category,
                            "answer": answer, "error": "judge: unparseable after retries"})
            continue

        refused = REFUSAL_MARKER in answer.lower()
        row = {
            "id": qid,
            "question": question,
            "category": category,
            "answer": answer,
            "refused": refused,
            "retrieved_sources": [c.get("filename") for c in citations],
            "faithfulness": scores["faithfulness"],
            "answer_correctness": scores["answer_correctness"],
            "context_relevance": scores["context_relevance"],
            "judge_reason": scores["reason"],
            "generation_ms": round(gen_ms, 1),
        }
        results.append(row)
        print(
            f"  [{qid}] faith={scores['faithfulness']:.2f} "
            f"correct={scores['answer_correctness']:.2f} "
            f"ctx={scores['context_relevance']:.2f} "
            f"({gen_ms:.0f}ms){' [refused]' if refused else ''} - {question[:45]}"
        )

    scored = [r for r in results if "error" not in r]
    errored = [r for r in results if "error" in r]

    def mean(key: str) -> float:
        vals = [r[key] for r in scored]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    report = {
        "config": {
            "owner_id": EVAL_OWNER_ID,
            "generator": "gemini (pipeline default)",
            "judge_model": JUDGE_MODEL,
            "rerank_threshold": os.getenv("RERANK_THRESHOLD", "-5.0 (default)"),
            "max_context_tokens": os.getenv("MAX_CONTEXT_TOKENS", "4000 (default)"),
            "reranker_model": os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2 (default)"),
        },
        "summary": {
            "total_questions": len(results),
            "scored": len(scored),
            "errored": len(errored),
            "avg_faithfulness": mean("faithfulness"),
            "avg_answer_correctness": mean("answer_correctness"),
            "avg_context_relevance": mean("context_relevance"),
        },
        "results": results,
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    s = report["summary"]
    print("\n" + "=" * 64)
    print("ANSWER-QUALITY SUMMARY")
    print("=" * 64)
    print(f"  Questions scored:      {s['scored']}/{s['total_questions']}  (errored: {s['errored']})")
    print(f"  Avg faithfulness:      {s['avg_faithfulness']:.3f}")
    print(f"  Avg answer correctness:{s['avg_answer_correctness']:.3f}")
    print(f"  Avg context relevance: {s['avg_context_relevance']:.3f}")
    print(f"\n  Report: {REPORT_PATH}")
    if errored:
        print(f"  NOTE: {len(errored)} question(s) errored and are excluded from averages.")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only evaluate the first N questions (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="fake generator and judge; no keys or network")
    args = ap.parse_args()
    asyncio.run(run(limit=args.limit, dry_run=args.dry_run))
