import os
import time
import uuid
import structlog
import secrets
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Security, HTTPException, status, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from app.api.routes import documents, chat, health, metrics, auth
from app.core.exceptions import ProcessingError, UnsupportedFormatError, DocumentTooLargeError, ScrapingError
from app.core.logging import setup_logging
from app.core.cache import init_cache, close_cache, increment_metric
from app.core.rate_limit import limiter
from app.core.telemetry import init_telemetry
from app.core.auth import require_role

# Initialize structlog first
setup_logging()
logger = structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Second Brain RAG...")
    await init_cache()
    
    from app.rag.vectorstore.chroma_store import get_stats
    try:
        stats = get_stats()
        logger.info("ChromaDB connected", stats=stats)
    except Exception as e:
        logger.error("ChromaDB connection failed", error=str(e))
            
    yield
    await close_cache()
    logger.info("Shutting down...")

app = FastAPI(title="Second Brain RAG API", lifespan=lifespan)

# Initialize OpenTelemetry tracing (auto-instruments FastAPI)
init_telemetry(app)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

API_KEY = os.getenv("API_KEY")
API_KEYS_RAW = os.getenv("API_KEYS", "")

# Build the set of valid API keys (supports multiple comma-separated keys)
_valid_keys: set = set()
if API_KEYS_RAW:
    _valid_keys = {k.strip() for k in API_KEYS_RAW.split(",") if k.strip()}
if API_KEY:
    _valid_keys.add(API_KEY)

if not _valid_keys:
    raise RuntimeError("No API keys configured. Set API_KEY or API_KEYS environment variable.")
if _valid_keys == {"default-secret-key-change-in-prod"} or _valid_keys == {"change-me-to-a-strong-secret"}:
    raise RuntimeError("API key(s) are still the default placeholder. Set strong secrets before starting.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate API Key"
        )
    # Constant-time comparison against each valid key
    if not any(secrets.compare_digest(api_key_header, k) for k in _valid_keys):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate API Key"
        )

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID", "Authorization"],
)

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000.0
    
    response.headers["X-Request-ID"] = request_id
    
    logger.info("HTTP Request",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms)
                
    try:
        await increment_metric("total_response_time_ms", duration_ms)
        await increment_metric("total_http_requests", 1)
    except Exception as e:
        logger.error("Failed to track metrics", error=str(e))
    
    return response

def build_rfc_7807(status: int, title: str, detail: str, req: Request):
    return JSONResponse(
        status_code=status,
        content={
            "type": "about:blank",
            "title": title,
            "status": status,
            "detail": detail,
            "instance": str(req.url)
        }
    )

@app.exception_handler(ProcessingError)
async def processing_exception_handler(request: Request, exc: ProcessingError):
    return build_rfc_7807(400, "Processing Error", str(exc), request)

@app.exception_handler(DocumentTooLargeError)
async def doc_large_exception_handler(request: Request, exc: DocumentTooLargeError):
    return build_rfc_7807(413, "Payload Too Large", str(exc), request)

@app.exception_handler(UnsupportedFormatError)
async def unsupported_format_exception_handler(request: Request, exc: UnsupportedFormatError):
    return build_rfc_7807(415, "Unsupported Media Type", str(exc), request)

@app.exception_handler(ScrapingError)
async def scraping_exception_handler(request: Request, exc: ScrapingError):
    return build_rfc_7807(502, "Bad Gateway", str(exc), request)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", error=str(exc))
    if hasattr(exc, "status_code"):
        return build_rfc_7807(exc.status_code, "HTTP Error", getattr(exc, "detail", str(exc)), request)
    return build_rfc_7807(500, "Internal Server Error", "An unexpected error occurred.", request)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(metrics.router, dependencies=[Depends(verify_api_key), Depends(require_role("admin"))])
app.include_router(documents.router, dependencies=[Depends(verify_api_key)])
app.include_router(chat.router, dependencies=[Depends(verify_api_key)])

@app.get("/")
async def root(request: Request):
    return {"message": "Welcome to Second Brain RAG API"}
