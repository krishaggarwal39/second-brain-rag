"""
Shared setup for the answer-quality evaluation harness.

Runs the RAG pipeline FULLY LOCALLY — no Docker, no running backend:
  * ChromaDB uses a local PersistentClient (CHROMA_PERSIST_DIR)
  * BM25 uses a local SQLite index (BM25_DATA_DIR)
  * Cache is in-memory (CACHE_BACKEND=memory)
  * OpenTelemetry disabled

IMPORTANT (zero-bug design): the real application always queries through the
owner-scoped path (`pipeline.ask(question, owner_id=str(user_id))`). This harness
ingests and queries with a FIXED owner_id so it measures exactly that path, not
the global (pre-isolation) retrieval path the older evaluation/custom_eval.py used.
"""

import os
import sys

# --- Local-mode environment (must be set BEFORE importing backend modules) ---
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.join(_EVAL_DIR, "..", "backend")

os.environ.setdefault("CACHE_BACKEND", "memory")
os.environ.setdefault("CHROMA_PERSIST_DIR", os.path.join(_EVAL_DIR, "answer_eval_chroma_data"))
os.environ.setdefault("BM25_DATA_DIR", os.path.join(_EVAL_DIR, "answer_eval_bm25_data"))
os.environ.setdefault("OTEL_ENABLED", "false")
# Prevent accidental connection to a real Chroma server: point the HTTP host at a
# non-resolvable name so get_client() falls back to the local PersistentClient fast.
os.environ.setdefault("CHROMA_HOST", "localhost")

# Local Neo4j instance overrides for running the eval outside Docker.
# backend/.env uses the docker-internal host (bolt://neo4j:7687); locally we talk to
# the container's published port. The password is read from the environment
# (EVAL_NEO4J_PASSWORD, falling back to the NEO4J_PASSWORD already in backend/.env) so
# no secret is hardcoded in source. Set EVAL_NEO4J_PASSWORD in your shell or .env.
os.environ.setdefault("NEO4J_URI", os.getenv("EVAL_NEO4J_URI", "bolt://localhost:7687"))
os.environ.setdefault("NEO4J_USER", os.getenv("EVAL_NEO4J_USER", "neo4j"))
_eval_neo4j_pw = os.getenv("EVAL_NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD")
if _eval_neo4j_pw:
    os.environ.setdefault("NEO4J_PASSWORD", _eval_neo4j_pw)

# Load real API keys from backend/.env (GOOGLE_API_KEY for the Gemini generator,
# GROQ_API_KEY for the judge) without overriding the local-mode vars set above.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_BACKEND_DIR, ".env"), override=False)
except Exception:
    pass

# Make the backend package importable.
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_BACKEND_DIR))

# Fixed identity for the evaluation tenant.
EVAL_OWNER_ID = "eval-user"

TEST_DOCS_DIR = os.path.join(_EVAL_DIR, "test_documents")
REPORTS_DIR = os.path.join(_EVAL_DIR, "reports")
# Gold set is selectable via GOLD_FILE env so the same harness can run against
# the easy synthetic set (gold_qa.json) or a harder real-doc set (gold_qa_well.json).
GOLD_PATH = os.path.join(_EVAL_DIR, os.getenv("GOLD_FILE", "gold_qa.json"))


def require_api_keys() -> None:
    """Fail loudly if the keys needed to run the harness are missing.

    The harness must never fabricate numbers; if it cannot generate or judge,
    it stops here with a clear message.
    """
    missing = []
    if not os.getenv("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY (answer generator / Gemini)")
    if not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY (LLM judge / Groq)")
    if missing:
        raise SystemExit(
            "ERROR: missing API keys required for the answer-quality eval:\n  - "
            + "\n  - ".join(missing)
            + "\nSet them in backend/.env and re-run. Refusing to fabricate scores."
        )
