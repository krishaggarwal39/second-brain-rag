# Second Brain RAG

A production-grade Retrieval-Augmented Generation system that turns your documents into a queryable knowledge base. Upload PDFs, DOCX files, or web pages — then ask questions and get accurate, cited answers powered by hybrid search and a live knowledge graph.

---

## Architecture

```
┌─────────────┐       ┌──────────────────────────────────────────────────────┐
│  React UI   │──────▶│  Nginx (reverse proxy, API key injection, SSE)       │
│  (Vite/TS)  │       └───────────────────────┬──────────────────────────────┘
└─────────────┘                               │
                                              ▼
                               ┌──────────────────────────┐
                               │   FastAPI Backend         │
                               │   (Gunicorn + Uvicorn)    │
                               └─────┬────────┬───────┬───┘
                                     │        │       │
                    ┌────────────────┘        │       └────────────────┐
                    ▼                          ▼                        ▼
          ┌─────────────────┐     ┌────────────────────┐    ┌──────────────┐
          │   ChromaDB      │     │   Neo4j            │    │   Redis      │
          │   (vectors)     │     │   (knowledge graph)│    │   (cache)    │
          └─────────────────┘     └────────────────────┘    └──────────────┘
```

**Query Flow:**
1. User question is embedded locally (all-MiniLM-L6-v2)
2. Parallel retrieval: Dense search (ChromaDB) + Sparse keyword search (BM25/SQLite FTS5) + Graph context (Neo4j)
3. Results merged via Reciprocal Rank Fusion, then reranked by a local Cross-Encoder (ms-marco-MiniLM-L-6-v2)
4. Token-budgeted prompt assembled with security directives
5. Streamed response from Gemini (primary) or Groq/LLaMa (fallback) via SSE

---

## Key Features

**Hybrid RAG Pipeline**
- Dense vector search (ChromaDB) + sparse keyword search (SQLite FTS5/BM25)
- Reciprocal Rank Fusion merges both result sets
- Local Cross-Encoder reranking for precision
- Parent-child chunking: small chunks for retrieval, full parent context for generation

**GraphRAG**
- LLM-extracted entities and relationships stored in Neo4j
- 1-hop graph traversal adds relational context at query time

**Multi-Provider LLM with Automatic Fallback**
- Primary: Google Gemini 2.5 Flash
- Fallback: Groq (LLaMa 3.3 70B)
- Automatic switch on 429 rate limits with cooldown-based recovery

**Security**
- Prompt injection defense: UUID-tagged context blocks, HTML escaping, security directives
- SSRF protection: DNS resolution → private IP blocking → curl `--resolve` pinning
- API key never in frontend bundle (Nginx injects server-side)
- Multi-key authentication with constant-time comparison
- Strict file size limits and document page caps

**Observability**
- OpenTelemetry tracing (FastAPI auto-instrumentation + custom pipeline spans)
- Structured JSON logging via structlog with API key redaction
- Metrics tracking: queries, tokens, cost, response times

**Resilience**
- Retry with exponential backoff on Neo4j and ChromaDB transient failures
- Semantic caching (Redis) with per-tenant key isolation
- Graceful degradation: graph/reranker failures don't crash retrieval

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Vite, TanStack React Query, Lucide Icons |
| Backend | Python 3.11, FastAPI, Gunicorn, async throughout |
| LLMs | Google Gemini (primary), Groq/LLaMa 3.3 (fallback) |
| Embeddings | Local SentenceTransformers (all-MiniLM-L6-v2) |
| Reranking | Local CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| Vector Store | ChromaDB |
| Graph DB | Neo4j 5 |
| Cache | Redis (with in-memory fallback) |
| Sparse Search | PostgreSQL full-text (compose default) or SQLite FTS5 |
| Observability | OpenTelemetry, structlog |
| Infrastructure | Docker Compose, Nginx |
| CI | GitHub Actions (lint + tests + RAG evaluation) |

---

## Getting Started

### Prerequisites
- Docker and Docker Compose
- A Google AI API key ([get one here](https://aistudio.google.com/app/apikey))
- (Optional) A Groq API key for fallback

### 1. Clone and configure

```bash
git clone https://github.com/krish-the-goat/second-brain-rag.git
cd second-brain-rag
cp .env.example .env
```

Set `GOOGLE_API_KEY` (and optionally `GROQ_API_KEY`) in `.env`. Keep the file private (`chmod 600 .env`).

### 2. Run with Docker Compose

Development (ports bound to `127.0.0.1`, clearly marked dev-only default passwords):

```bash
docker compose up -d --build
```

Production (every secret is required; compose refuses to start with a missing one):

```bash
./deploy.sh --generate-secrets   # fills missing secrets in .env with random hex, then deploys
```

`deploy.sh` validates `.env` by key name, runs `docker compose config`, starts the stack and waits for `/api/health/ready`.
The UI is at `http://localhost:5173` (dev) or `http://localhost` (prod).

The stack runs: `frontend` (nginx, non-root), `backend` (FastAPI), `worker` (Celery ingestion), `postgres` (sparse index and user accounts), `chroma` (vectors, token auth enforced), `neo4j` (graph), `redis` (cache, job status, rate limits). API and worker share the `uploads` volume.

### HTTPS

Plain HTTP must not be exposed publicly. The prod compose file has an optional Caddy front door that gets a certificate automatically:

```bash
# .env: DOMAIN=rag.example.com  ACME_EMAIL=ops@example.com  FRONTEND_PORT=127.0.0.1:8080  REAL_IP_FROM=172.16.0.0/12
docker compose -f docker-compose.prod.yml --profile tls up -d
```

nginx sends HSTS only when the request arrived over HTTPS.

### First user and admin

Register in the UI, then promote the account (see "Admin roles" below):

```bash
docker compose -f docker-compose.prod.yml exec backend python -m app.cli promote you@example.com
```

### 3. Upload documents

Use the sidebar to upload PDFs/DOCX files or paste a web URL. Documents are chunked, embedded, and indexed automatically. Uploaded files are staged in `UPLOAD_DIR` (default `<tempdir>/second_brain_uploads`, mode 0700) and shared with the background worker.

### 4. Ask questions

Type a question in the chat. The system retrieves relevant context, augments it with graph relationships, and streams the answer with source citations.

---

## Development

### Backend tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

### Frontend tests

```bash
cd frontend
npm install
npm test
```

### Linting

```bash
cd backend && ruff check app/
cd frontend && npm run lint
```

---

## Configuration

All configuration is via environment variables. See [`.env.example`](.env.example) for the full list with descriptions.

Key tuning parameters:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 2000 | Parent chunk size (chars) |
| `CHUNK_OVERLAP` | 200 | Overlap between parent chunks |
| `RERANK_THRESHOLD` | -5.0 | Cross-encoder score cutoff |
| `MAX_CONTEXT_TOKENS` | 4000 | Token budget for LLM context |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | SentenceTransformer model |
| `RERANKER_MODEL` | cross-encoder/ms-marco-MiniLM-L-6-v2 | Reranker model |
| `GEMINI_MODEL` | gemini-2.5-flash | Primary LLM |
| `GROQ_MODEL` | llama-3.3-70b-versatile | Fallback LLM |

---

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/routes/        # FastAPI route handlers
│   │   ├── core/              # Auth, cache, logging, telemetry, resilience
│   │   ├── rag/
│   │   │   ├── chunkers/      # Parent-child recursive chunking
│   │   │   ├── embeddings/    # Local SentenceTransformer embedder
│   │   │   ├── graph/         # Neo4j manager, graph extraction & retrieval
│   │   │   ├── loaders/       # PDF, DOCX, Web (SSRF-safe) loaders
│   │   │   ├── retrievers/    # Hybrid search with RRF + cross-encoder
│   │   │   └── vectorstore/   # ChromaDB + BM25 (SQLite FTS5) stores
│   │   │   ├── context_engineering.py  # Token-budgeted prompt builder
│   │   │   ├── ingestion.py   # Document processing pipeline
│   │   │   └── pipeline.py    # Main RAG orchestrator
│   │   └── main.py            # FastAPI app entry point
│   ├── tests/                 # pytest suite (344 tests; see VERIFICATION_LOG.md)
│   ├── requirements.txt       # Pinned production deps
│   └── requirements-dev.txt   # Dev/test deps
├── frontend/
│   ├── src/
│   │   ├── components/        # React components + tests
│   │   ├── services/          # Axios API client
│   │   └── App.tsx            # Main layout
│   └── package.json           # Pinned deps
├── evaluation/                # RAG quality evaluation pipeline
│   ├── test_queries.json      # 22 test queries
│   ├── generate_test_docs.py  # Synthetic document generator
│   ├── ingest_test_docs.py    # Local ingestion script
│   └── custom_eval.py         # Retrieval quality metrics
├── docker-compose.yml         # Dev stack
├── docker-compose.prod.yml    # Production stack with healthchecks
└── .github/workflows/ci.yml   # CI: lint + test + eval
```

---

## Upgrading

### GraphRAG Tenant Isolation & Purging Legacy Graph Data
Starting with the tenant isolation update (R-01), all Neo4j graph nodes and relationships carry an `owner_id`. Any existing graph data without an `owner_id` is considered legacy and is automatically excluded from search and retrieval.

To permanently delete legacy unowned nodes and relationships from Neo4j, run the idempotent maintenance helper:

```python
from app.rag.graph.neo4j_manager import purge_legacy_graph

deleted_count = purge_legacy_graph()
print(f"Purged {deleted_count} legacy graph nodes")
```

This helper is safe to run multiple times and does not run automatically on startup.

### Admin roles

Registration always creates a normal `user`, and emails are matched case-insensitively. The role stored in the database decides access on every request, so a change applies immediately. Register the account first, then promote it from inside the backend container:

```bash
python -m app.cli promote admin@example.com
python -m app.cli demote admin@example.com
python -m app.cli promote-from-env   # promotes already-registered users listed in ADMIN_EMAILS
```

`ALLOW_ADMIN_SELF_REGISTER=true` restores the old behaviour (ADMIN_EMAILS applied on register/login). Use it for local development only. There is no email verification, so whoever registers a listed address first would become admin.

## Security model

- The JWT is the per-user boundary. All documents, vectors, sparse index rows, cache entries and graph data are scoped by the user id from the token.
- `X-API-Key` is a coarse shared gate. In the Docker setup nginx injects it for every caller, so it is not per-user security.
- Auth endpoints are rate limited per IP (`AUTH_RATE_LIMIT`). Repeated failed logins for one email trigger a temporary lockout. Other limits are keyed per user when a valid token is present.
- Behind a reverse proxy, set `TRUSTED_PROXY_CIDRS` so client IPs come from `X-Forwarded-For`. The header is ignored from any other peer.

---

## Limitations & Future Work

- **No email verification**: registration is open; admins are promoted out of band (`python -m app.cli`).
- **Answer-quality evaluation needs paid keys**: CI runs the keyword-overlap retrieval eval and a dry run of `answer_eval.py`; the real LLM-as-judge run is manual.
- **Chroma stays on 0.5.x**: Chroma 1.x OSS has no built-in server auth, so the server and the HTTP-only client are pinned to 0.5.23 together.

### Scaling the sparse store

Both compose stacks use the PostgreSQL backend by default. The SQLite FTS5 store (handy outside Docker) uses file-level locking and must run with single-worker Gunicorn. To use Postgres outside Docker, set:

```bash
SPARSE_BACKEND=postgres
DATABASE_URL=postgresql://postgres:postgrespassword@postgres:5432/second_brain
GUNICORN_WORKERS=4
```

The Postgres store uses a `tsvector` GIN index with OR-semantics `to_tsquery` (the same matching as SQLite) and a fork-aware connection pool safe for multi-process Gunicorn.

Potential improvements:
- Multi-modal document parsing (images, tables as structured data)
- Streaming graph updates with change-data-capture
- Elasticsearch sparse backend for cross-node sharding

---

## Contributing

PRs welcome. Please ensure:
1. `ruff check` passes with no errors
2. All existing tests pass (`pytest` / `npm test`)
3. New features include corresponding tests

---

## License

MIT
