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
| Sparse Search | SQLite FTS5 |
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

Edit `.env` with your API keys and strong passwords for databases.

### 2. Run with Docker Compose

```bash
docker-compose up -d --build
```

The UI will be available at `http://localhost:5173` (dev) or `http://localhost:80` (production compose).

### 3. Upload documents

Use the sidebar to upload PDFs/DOCX files or paste a web URL. Documents are chunked, embedded, and indexed automatically.

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
│   ├── tests/                 # pytest suite (~76 tests)
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

## Limitations & Future Work

- **Single-tenant**: Multi-key auth provides basic client separation but not full RBAC or user-scoped document isolation.
- **Evaluation is retrieval-only**: The eval pipeline tests context quality but not end-to-end answer quality (would need LLM-as-Judge with API costs).

### Scaling the sparse store

The default SQLite FTS5 BM25 store uses file-level locking and must run with single-worker Gunicorn. A PostgreSQL full-text-search backend is also included for horizontal scaling across multiple workers. To enable it, set in your `.env`:

```bash
SPARSE_BACKEND=postgres
DATABASE_URL=postgresql://postgres:postgrespassword@postgres:5432/second_brain
GUNICORN_WORKERS=4
```

The Docker Compose stack already provisions a `postgres` service. The Postgres store uses a `tsvector` GIN index with `websearch_to_tsquery` ranking and a fork-aware connection pool safe for multi-process Gunicorn.

Potential improvements:
- Role-based access control with JWT
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
