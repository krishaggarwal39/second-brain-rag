FROM python:3.11.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

COPY requirements.txt .
# CPU-only torch on every architecture (the PyPI aarch64 wheels pull in several GB of
# CUDA libraries the app never uses).
RUN pip install --default-timeout=1000 --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Pre-fetch the embedding and reranker models at build time so the first request
# does not download them (and the container does not need outbound HF access).
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}'); CrossEncoder('${RERANKER_MODEL}')" \
    && chown -R app:app /opt/models

# Models are baked in; never call the Hugging Face Hub at runtime. Changing
# EMBEDDING_MODEL / RERANKER_MODEL requires rebuilding with the matching build args.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY --chown=app:app . .
RUN mkdir -p /app/bm25_data /app/auth_data /data/uploads \
    && chown -R app:app /app/bm25_data /app/auth_data /data

USER app

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# One worker by default: the SQLite sparse store is unsafe across processes.
# With SPARSE_BACKEND=postgres (the compose default) GUNICORN_WORKERS can be raised.
# --timeout is the worker heartbeat timeout; with async Uvicorn workers it does not
# cut off long SSE streams. FORWARDED_ALLOW_IPS controls uvicorn's proxy-header trust;
# per-client rate limiting reads X-Forwarded-For itself via TRUSTED_PROXY_CIDRS.
CMD ["sh", "-c", "exec gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w ${GUNICORN_WORKERS:-1} --bind 0.0.0.0:8000 --timeout 120 --graceful-timeout 30 --max-requests 1000 --max-requests-jitter 100 --forwarded-allow-ips ${FORWARDED_ALLOW_IPS:-127.0.0.1}"]
