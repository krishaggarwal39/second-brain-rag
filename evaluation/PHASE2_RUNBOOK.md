# Phase 2 Tuning Runbook — improving answer accuracy against the locked baseline

## >>> PAUSED — RESUME WITH A HARDER DOCUMENT <<<

STATUS: Phase 2 tuning paused by choice. One proven win shipped (child_size=250).
Tuning on the current well_deliverability.pdf has hit diminishing returns — the system
already answers well when it has context (correctness ~0.88), so the remaining levers
don't move accuracy much on THIS doc.

TO RESUME (bigger gains need a harder/longer document):
1. Drop a new document into `evaluation/test_documents/` and add a gold Q&A set
   `evaluation/gold_qa_<name>.json` (copy gold_qa_well.json's format: question,
   reference_answer, source_doc, category).
2. Ingest: `python ingest_doc.py test_documents/<file> --name <file>`
3. Make sure Gemini (generator) + OpenRouter (judge) quotas are healthy
   (both error -> see the quota notes below; add backup keys if needed).
4. Establish a baseline (EVAL_CONFIG_TAG=baseline_<name>), then A/B each lever with a
   fresh tag, comparing on COMMON scored questions.

CURRENT BEST CONFIG (already the default in code): child_size=250 + L-6 reranker.
UNCOMMITTED code changes to keep: `recursive_chunker.py` (child 250) and
`docker-compose.prod.yml` (prod worker). Decide whether to commit when ready.
REMINDER: rotate the Groq + OpenRouter API keys pasted in chat.

---

## >>> EXPERIMENT RESULT: grounding-prompt v1 — REVERTED (measured, failed) <<<

Tested the strict grounding prompt against the real-doc baseline (Groq judge, full 27/27):
| Metric | Baseline | Grounding v1 | Verdict |
|---|---|---|---|
| Faithfulness | 0.800 | 0.787 | flat |
| Answer correctness | 0.865 | **0.698** | REGRESSED |
| Context relevance | 0.812 | 0.783 | slightly down |

Why it failed: the "add NO outside knowledge" wording was too blunt. On terse slide
bullets the model started REFUSING in-scope questions it could actually answer
(w09 transient flow 0.9->0.0, w18 flow efficiency 0.9->0.0 both wrongly refused).
Net: worse. REVERTED `backend/app/rag/context_engineering.py` to original (empty diff).
Saved the failed run at `reports/well_grounding_v1.json` for the record.

LESSON for the next attempt: the goal is "don't INVENT facts" WITHOUT "refuse when the
answer is present but terse". A better v2 prompt would (a) allow answering from brief
context, (b) forbid only fabricated specifics, (c) NOT add a hard refuse instruction.
Measure the same way before keeping.

---

## >>> PHASE 2 RESULTS (session 4) — FIRST MEASURED WIN <<<

Generator = Gemini, Judge = OpenRouter llama-3.3-70b (same as locked baseline), real doc.

| Config | Faithfulness | Answer correctness | Context relevance | Verdict |
|---|:---:|:---:|:---:|---|
| baseline (child 400) | 0.956 | 0.896 | 0.896 | reference |
| RETRIEVAL_MULTIPLIER=5 | 0.911 | 0.830 | 0.885 | REJECTED (worse) |
| child 300 | 0.952 | 0.878 | 0.907 | partial gain |
| **child 250** | **0.956** | 0.881 | **0.941** | **KEPT** |

DECISION: set default `CHILD_CHUNK_SIZE=250` in `backend/app/rag/chunkers/recursive_chunker.py`.
Rationale: smaller child chunks consistently raised retrieval context-relevance
(0.896 -> 0.907 -> 0.941 as 400 -> 300 -> 250) by isolating a topic per chunk, fixing
retrieval misses (w04 ctx 0.20->0.50, w05 0.70->1.00, w06 0.60->0.80, w18 0.80->1.00).
Faithfulness unchanged (0.956); answer_correctness -0.015 (within judge noise). Net:
clear improvement on the foundation metric (retrieval) at negligible cost.
Saved runs: reports/exp_retmult5.json, reports/exp_child250.json.

### Reranker upgrade — TESTED, REJECTED
Swapped reranker to `cross-encoder/ms-marco-MiniLM-L-12-v2` on top of child250.
Fair comparison on the 22 questions common to all runs (Gemini quota erroring the tail):
| Config | Faith | Correct | Ctx |
|---|:---:|:---:|:---:|
| child250 (L-6, current best) | 0.950 | 0.859 | 0.927 |
| child250 + L-12 reranker | 0.955 | 0.850 | 0.914 |
L-12 is 2x the size/slower for NO gain (ctx and correct slightly DOWN, faith within noise).
REJECTED. Default reranker stays `ms-marco-MiniLM-L-6-v2`. Saved: reports/exp_child250_rerankL12.json.

### Current best config: child_size=250 + L-6 reranker (default)
Next candidate lever (untested): a GENTLER grounding prompt than the strict v1 that
over-refused (session 1). Also worth: bge-reranker-base (different family) if desired,
but reranker gains look unlikely on this doc.

NOTE: Gemini daily quota is near its limit again — runs error on the last ~5 questions.
Compare on COMMON questions across runs (as above) until quota resets, or add a backup
generator key.

---

## >>> LOCKED BASELINE (session 3) — USE THIS AS THE REFERENCE <<<

Clean, complete, reproducible real-doc baseline. Judge = OpenRouter
`meta-llama/llama-3.3-70b-instruct` (fresh key). Default pipeline config.
File: `reports/well_baseline_or.json`.

| Metric | Score |
|---|---|
| Faithfulness | 0.956 |
| Answer correctness | 0.896 |
| Context relevance | 0.896 |
27/27 scored, 0 errors. Verified: re-running reproduces 0.956/0.896/0.896 EXACTLY
from cache with zero API calls.

IMPORTANT — judge model changes the numbers: the earlier Groq `gpt-oss` judge was much
harsher (gave ~0.80 faithfulness on the SAME system). So ONLY compare runs judged by the
SAME judge. This OpenRouter-judged baseline is the reference for all Phase 2 experiments.

### Harness is now outage-robust (session 3 additions, answer_eval.py)
- ANSWER cache (`reports/.answer_cache.json`), keyed by (EVAL_CONFIG_TAG, question):
  a run reuses previously-generated answers, so generator (Gemini) 429s don't block a run.
  **Set a NEW EVAL_CONFIG_TAG whenever you change pipeline config** (chunking/retrieval/
  prompt) so answers regenerate; reuse the same tag to re-judge without regenerating.
- JUDGE cache (`reports/.judge_cache.json`), keyed by (judge model, question, answer):
  never re-charges the judge for an already-scored answer.
- Provider error placeholders are excluded from scoring (not quality signals).

### How to run a Phase 2 experiment cleanly
```bash
cd evaluation
# 1. change ONE knob (e.g. edit chunker, or set an env var), then pick a NEW tag:
EVAL_CONFIG_TAG=chunk_v2 CHILD_CHUNK_SIZE=250 GOLD_FILE=gold_qa_well.json \
  JUDGE_PROVIDER=openrouter python answer_eval.py
# (if you changed chunking, re-ingest first: rm -rf answer_eval_chroma_data answer_eval_bm25_data; python ingest_doc.py test_documents/well_deliverability.pdf)
# 2. compare to baseline:
python - <<'PY'
import json
b=json.load(open('reports/well_baseline_or.json'))['summary']
c=json.load(open('reports/answer_quality_report.json'))['summary']
for k in ('avg_faithfulness','avg_answer_correctness','avg_context_relevance'):
    print(f"{k}: {b[k]:.3f} -> {c[k]:.3f} ({c[k]-b[k]:+.3f})")
PY
```

### Known real finding to note (not blocking)
When the generator (Gemini) is FULLY quota-exhausted, the pipeline's Groq fallback does
not reliably save the request — every question returned an error placeholder. The app
fallback is robust to intermittent 429s but not to a fully-down primary. Candidate real
improvement for the app itself (separate from eval).

---

## >>> HARNESS FIX + QUOTA REALITY (session 2) <<<

- FIXED a real harness bug: the eval now EXCLUDES provider error placeholders
  ("API Error (429)", "Check server logs.", etc.) from scoring instead of letting the
  judge score them ~0 and silently poison the average. Contaminated answers now show as
  "errored" and are left out of aggregates. (answer_eval.py)
- QUOTA REALITY: OpenRouter is permanently exhausted (total-limit 403). Only Groq works,
  but when the GENERATOR (Gemini) rate-limits mid-run it falls back to Groq and collides
  with the Groq JUDGE — so each run scores a DIFFERENT subset (23-27/27) and aggregates
  wobble (0.67-0.80) for infrastructure reasons, NOT quality. Full clean comparisons are
  not reliably possible on current free quotas.
- RECOMMENDATION to get clean numbers: either (a) top up OpenRouter so the judge has its
  own quota again (best), or (b) run each config several times across the day so the judge
  cache fills all 27 and the generator-429 answers eventually come back real. Do NOT grind
  in a single sitting.
- Trustworthy numbers so far: `reports/well_baseline.json` (the original clean 24/27 real-doc
  baseline, ~0.80/0.865/0.81) and `reports/well_grounding_v1.json` (the failed grounding
  experiment). Everything measured after quota got tight is subset-contaminated — ignore.

---

## >>> WHERE WE ARE / RESUME (read this first) <<<

We now have a REAL, discriminating test document and a real baseline. Status:

- Real doc ingested: `evaluation/test_documents/well_deliverability.pdf` (39 slides).
- Real gold set: `evaluation/gold_qa_well.json` (25 in-scope + 2 out-of-scope).
- Real-doc BASELINE (saved `reports/well_baseline.json`), Groq judge `gpt-oss-120b`,
  default chunking, grounding prompt NOT yet changed:
    - faithfulness ~0.80, answer_correctness ~0.865, context_relevance ~0.81
- KEY FINDING: the bottleneck is FAITHFULNESS. Retrieval finds the right slides, but
  the generator ADDS facts from its own training knowledge not in the slides
  (e.g. Darcy's-law questions scored 0.20 on faithfulness).
- FIX APPLIED (accepted, awaiting verification): strengthened the grounding directive
  in `backend/app/rag/context_engineering.py` (`build_dynamic_prompt`) to "answer ONLY
  from context, add no outside facts". This change is KEPT in the code.

### BLOCKER: both free judge quotas are exhausted
- OpenRouter: total credit limit reached (403; needs top-up, won't reset on its own).
- Groq: 8K tokens/min free tier; a full 27-question run trips 429s partway.
Do NOT grind against the limit. The judge now does only 2 gentle retries then skips;
a per-(question,answer,judge) cache in `reports/.judge_cache.json` preserves every
score already obtained, so resuming only spends quota on what's left.

### TO VERIFY THE GROUNDING FIX (do this when quota resets, ~1 day):
```bash
cd evaluation
# make sure the real doc is ingested (idempotent):
python ingest_doc.py test_documents/well_deliverability.pdf
# run the SAME config as the baseline, now WITH the grounding fix in place:
GOLD_FILE=gold_qa_well.json JUDGE_PROVIDER=groq EVAL_PACING_SECONDS=10 python answer_eval.py
```
Then compare the new summary to `reports/well_baseline.json`:
- If avg_faithfulness RISES meaningfully (e.g. 0.80 -> 0.88+) WITHOUT answer_correctness
  dropping, the grounding fix WORKED — keep it, record the numbers below.
- If faithfulness does NOT improve or correctness drops, revert the change in
  `backend/app/rag/context_engineering.py`.
Tip: if Groq still rate-limits, run it a few times across the day — the judge cache
fills in the missing questions each time until all 27 are scored.

---

This runbook drives the accuracy-tuning loop. Each step changes ONE thing, re-runs
the eval, and compares against the locked baseline. It is designed to be driven
mechanically (including by a terminal CLI agent) to conserve credits — the expensive
reasoning (diagnosis) is already done and recorded below.

## Ground truth / setup (already done in Phase 1)

- Harness lives in `evaluation/`: `eval_common.py`, `ingest_gold.py`, `answer_eval.py`, `gold_qa.json`.
- Runs FULLY LOCALLY (no Docker). Chroma = local PersistentClient, BM25 = local SQLite, cache = in-memory.
- Generator = the real app pipeline (`pipeline.ask(owner_id="eval-user")`), Gemini primary.
- Judge = OpenRouter `meta-llama/llama-3.3-70b-instruct` (separate quota; different family → no self-grading bias).
- Keys are in `backend/.env` (gitignored). ROTATE the pasted Groq/OpenRouter keys when done.

### Locked baseline (reproducible, 2 runs within ±0.02) — `reports/baseline_locked.json`
| Metric | Score |
|---|---|
| Faithfulness | ~0.90 |
| Answer correctness | 0.96 |
| Context relevance | ~0.81 |

## Commands

Re-run the full eval (index already built; only re-ingest if you change chunking — see step 1):
```bash
cd evaluation
python answer_eval.py                      # full gold set -> reports/answer_quality_report.json
python answer_eval.py --limit 3            # quick smoke test
```
Override a knob for one run without editing .env (shell env wins; eval_common loads .env with override=False):
```bash
RERANK_THRESHOLD=-5.0 python answer_eval.py
```
Compare a result to the baseline:
```bash
python - <<'PY'
import json
base=json.load(open('reports/baseline_locked.json'))['summary']
cur =json.load(open('reports/answer_quality_report.json'))['summary']
for k in ('avg_faithfulness','avg_answer_correctness','avg_context_relevance'):
    print(f"{k}: {base[k]:.3f} -> {cur[k]:.3f}  (delta {cur[k]-base[k]:+.3f})")
PY
```

## DIAGNOSIS — CORRECTED (read this; the first guess was wrong)

Honest trail of what we learned, so you don't repeat the dead ends:

1. First guess: RERANK_THRESHOLD -11.0 was letting noise through. **WRONG** — changing it
   to -5.0 moved aggregates within noise (0.81 -> 0.83). Not the lever.
2. Second guess: the chunker severs section headings from bodies (child 400, char-wise,
   no structural separators). The raw CHILD chunks *are* fragmented (verified by inspecting
   `hybrid_search` output). BUT...
3. **The real picture:** the retriever hydrates the full PARENT chunk into the LLM context
   (`hybrid_retriever.py` sets `parent_content` from the `parent:{id}` cache, and
   `context_engineering.build_dynamic_prompt` prefers `parent_content`). On these tiny
   3-doc synthetic files each doc is ONE ~1600-char parent, so the whole document reaches
   the LLM regardless of child fragmentation. Deterministic evals confirm retrieval is
   already saturated on this set:
     - `retrieval_eval.py` (source-doc): 22/22 hit, MRR 1.000 (only 3 docs -> trivially easy).
     - `chunk_eval.py` (keyword recall in retrieved text): 0.989, answer-chunk MRR 1.000.
4. So the low LLM-judge scores in the FIRST full runs (g04-g22 at 0.00) were mostly
   **provider failures, not retrieval failures**: the decommissioned Groq fallback model
   returned "API Error (404)" strings that the judge correctly scored 0. Once generation
   ran clean (OpenRouter judge, Gemini generator), faithfulness was already ~0.90.

### CONCLUSION
On the CURRENT synthetic eval set, retrieval/chunking is NOT a measurable bottleneck —
the set is too easy to discriminate between strategies. The structural-chunking change
in `recursive_chunker.py` (structural separators + env-driven CHILD_CHUNK_SIZE) is sound,
backward-compatible practice but is **NOT YET PROVEN to lift answer quality here**.

### THE ACTUAL NEXT STEP (highest value)
Build a HARDER gold set from REAL / longer multi-section documents (a textbook chapter,
paper, or long report) so a single big doc produces MANY parents and retrieval must find
the RIGHT passage among many. Only then can the eval tell chunking/reranking strategies
apart. Add the doc under `sample_docs/` or `evaluation/test_documents/`, extend
`gold_qa.json` with real Q&A + source, re-ingest, and re-measure. (User will provide a
real document when LLM quotas reset.)

### Remaining real behavior gap (not quota-dependent to confirm)
- g24 (out-of-scope "tell me a joke about cats"): the pipeline ANSWERED instead of
  refusing. Worth fixing via the refusal directive in `context_engineering.build_dynamic_prompt`
  (step 5 below), and confirmable once the judge quota resets.

## Tuning steps, in PRIORITY ORDER (highest expected impact first)

Each step: make the change, re-run, compare, KEEP only if context_relevance/faithfulness
improve with no regression in answer_correctness. Record the number in the table at the bottom.

### Step 1 — Structural / smaller chunking (HIGHEST IMPACT; addresses root cause)
Change in `backend/app/rag/chunkers/recursive_chunker.py`:
- Add explicit split separators so sections/paragraphs are respected, e.g. pass
  `separators=["\n\n", "\n", ". ", " ", ""]` to the child `RecursiveCharacterTextSplitter`.
- Reduce `child_size` (e.g. 400 -> 220-280) and make it env-driven (`CHILD_CHUNK_SIZE`),
  so a single topic/section tends to land in its own child with its heading.
- Optionally reduce parent `CHUNK_SIZE` so multi-topic docs produce multiple parents.
IMPORTANT: chunking changes require RE-INGESTION before re-running the eval:
```bash
rm -rf evaluation/answer_eval_chroma_data evaluation/answer_eval_bm25_data
cd evaluation && python ingest_gold.py && python answer_eval.py
```
Watch g13/g16/g09 context_relevance specifically.

### Step 2 — Increase retrieval candidate pool / top_k
`backend/app/rag/retrievers/hybrid_retriever.py` uses `fetch_k = top_k * RETRIEVAL_MULTIPLIER`
(RETRIEVAL_MULTIPLIER default 3). Try `RETRIEVAL_MULTIPLIER=5` (env) so the reranker sees
more candidates and the right chunk isn't lost before reranking. Re-run (no re-ingest needed).

### Step 3 — Upgrade the reranker model
`RERANKER_MODEL` default `cross-encoder/ms-marco-MiniLM-L-6-v2`. Try the larger
`cross-encoder/ms-marco-MiniLM-L-12-v2` or `BAAI/bge-reranker-base` (downloads once; env):
```bash
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2 python answer_eval.py
```
No re-ingest needed. Keep only if it lifts context_relevance without hurting latency unacceptably.

### Step 4 — Re-tune RERANK_THRESHOLD after chunking is fixed
Once chunks are clean, re-sweep the threshold (it was a no-op on bad chunks but may matter
on good ones): try -5.0, -3.0, -2.0. `RERANK_THRESHOLD=-3.0 python answer_eval.py`.
Then set the winning value in `backend/.env` (currently -11.0) and `.env.example`.

### Step 5 — Refusal / citation-grounding prompt (fixes g16 faithfulness + g24)
`backend/app/rag/context_engineering.py build_dynamic_prompt`: strengthen the directive so
the model (a) answers ONLY from context and explicitly refuses when context lacks the answer
(fixes g16 answering from parametric knowledge), and (b) refuses clearly off-topic requests
(fixes g24). Re-run; watch g16 faithfulness and g24 all-three.

## Results log

### LLM-judge (OpenRouter judge, Gemini generator) — baseline captured before quotas ran out
| Step | Change | Faithfulness | Answer-correct | Context-rel | Keep? |
|------|--------|--------------|----------------|-------------|-------|
| baseline | threshold -11.0, child 400 charwise | ~0.90 | 0.96 | ~0.81 | — |
| threshold probe | RERANK_THRESHOLD=-5.0 | 0.88 | 0.96 | 0.83 | no (within noise) |

### Deterministic (no-API) retrieval metrics — current synthetic set
| Config | source_hit@5 | source MRR | keyword_recall@5 | answer_chunk_MRR |
|--------|-------------|-----------|------------------|------------------|
| child 400 char-wise (old) | 1.000 | 1.000 | 0.989 | 1.000 |

Interpretation: retrieval is already saturated on this set, so chunking/reranker/threshold
changes cannot be distinguished here. Re-measure on a harder/real-doc gold set.

### To resume when LLM quotas reset
1. Re-point the judge: either top up OpenRouter credit (JUDGE_PROVIDER=openrouter) or use
   JUDGE_PROVIDER=groq with a model from your Groq limits screen (e.g. openai/gpt-oss-120b).
   Note Groq free tier is 8K TPM — if the GENERATOR also falls back to Groq, judge + generator
   contend; prefer OpenRouter for the judge, or pace heavily (EVAL_PACING_SECONDS=12+).
2. Add a REAL document + real Q&A to make the eval discriminating (see "THE ACTUAL NEXT STEP").
3. Re-establish a baseline on the new set, THEN run steps 1-5 below, one knob at a time.

| Step | Change | Faithfulness | Answer-correct | Context-rel | Keep? |
|------|--------|--------------|----------------|-------------|-------|
| 1 | structural + smaller chunks (CHILD_CHUNK_SIZE) | | | | unproven on easy set |
| 2 | RETRIEVAL_MULTIPLIER=5 | | | | |
| 3 | reranker L-12 / bge | | | | |
| 4 | re-tune threshold | | | | |
| 5 | refusal/grounding prompt (fixes g24) | | | | |

## Guardrails
- Change ONE knob per run; attribute each delta to one cause.
- answer_correctness must not regress. If a change lifts context_relevance but tanks
  correctness, reject it.
- Re-ingest ONLY after chunking changes (steps 1); other steps are query-time only.
- The judge is non-deterministic within ±0.02; treat deltas smaller than that as noise.
- When done: set winning values in `backend/.env` + `.env.example`, update the README
  benchmark table, and ROTATE the API keys.
```

## Latency baseline

First production-readiness step: measure WHERE query latency goes, per sub-stage, so
optimization is data-driven. This section is MEASUREMENT ONLY — nothing was optimized.

### How it was measured
- Instrumented `backend/app/rag/retrievers/hybrid_retriever.py` with `time.perf_counter()`
  around each sub-stage (timing-only; retrieval results are unchanged — verified below).
  Timings are attached to the existing OTEL span and exposed via
  `get_last_hybrid_timings()`.
- New harness `evaluation/latency_eval.py` runs fully locally (reuses `eval_common`'s env
  wiring: in-memory cache, local Chroma/BM25, OTEL off), over all 25 in-scope gold
  questions in `gold_qa_well.json`, owner `eval-user`.
- Models (embedder + CrossEncoder) load ONCE; the cold first query is recorded separately
  and EXCLUDED from the warm per-query aggregates.
- Run it: `GOLD_FILE=gold_qa_well.json python evaluation/latency_eval.py --repeats 3`
- Full numbers: `evaluation/reports/latency_report.json`. Run evidence (two+ runs,
  before/after retrieval-quality check): `evaluation/reports/latency_run_log.md`.

### Measured numbers (warm, median over 25 questions x 3 repeats; stable across runs)

| Sub-stage | median (ms) | p95 (ms) | share of hybrid total |
|-----------|-------------|----------|-----------------------|
| embedding (query encode) | 10.5 | 13.9 | ~13% |
| dense (ChromaDB) | 4.8 | 7.1 | ~6% |
| sparse (BM25) | 0.4 | 0.8 | <1% |
| RRF + metadata hydration | 2.0 | 3.0 | ~2% |
| **CrossEncoder rerank** | **62.8** | **158.0** | **~78%** |
| parent hydration (cache) | 0.03 | 0.04 | <1% |
| **hybrid total** | **81.0** | **175.2** | 100% |

- Cold first query (one-time model load, excluded above): ~4.2 s wall, almost all of it
  the embedder + CrossEncoder load — a startup cost, not a per-query cost.
- Graph retrieval (timed separately, runs in PARALLEL with hybrid_search in production):
  ~1.0–3.8 s per call, because entity extraction calls an external LLM. It is
  API/quota-bound, not local compute. Neo4j was not running locally, so the graph DB
  lookup itself was not exercised; the latency observed is the LLM entity-extraction round
  trip.
- Generation (full `pipeline.ask()`, external Gemini/Groq): measured opportunistically on a
  3-question sample while quota was available — median ~1.08 s, p95 ~1.53 s. Variable and
  external; not the local-retrieval bottleneck. The harness SKIPS (never fabricates)
  generation when quota is unavailable.

### Verdict — dominant sub-stage and the optimization target
**The CPU CrossEncoder rerank dominates local retrieval latency: median ~63 ms, ~78% of
the hybrid total of ~81 ms, and the single largest contributor to tail latency
(p95 ~158 ms).** Everything else (embedding ~10 ms, dense ~5 ms, BM25/fusion/hydration
combined <3 ms) is minor by comparison. The hypothesis held.

**Highest-value optimization target: the CrossEncoder rerank step.** Candidate directions
for the NEXT step (optimization — not done here), to be A/B'd against this baseline and
against retrieval quality (`retrieval_eval.py` must stay 25/25, MRR 1.0):
- Rerank fewer candidates (`RETRIEVAL_MULTIPLIER` is 3 -> fetch_k = top_k*3 = 15 pairs per
  query; reducing it cuts rerank work linearly).
- A lighter / quantized / ONNX-optimized cross-encoder, or batching tuning.
- Cache rerank scores for repeated (query, chunk) pairs.
- Consider skipping rerank when fused candidates are already well-separated.

Note on end-to-end: in production, graph retrieval and the LLM generation call dwarf the
~81 ms local retrieval in absolute terms, BUT graph runs in parallel and generation is
external API latency. Of the work we CONTROL locally and can optimize in code, the
CrossEncoder rerank is the clear target.

