"""
FastAPI application — the main API entry point.

API design principles:
- Async throughout: never block the event loop
- Fail fast: startup validates all configuration and dependencies
- Graceful degradation: if cache fails, still serve requests
- Request IDs: every request gets a UUID for tracing
- Structured error responses: always return JSON, never HTML
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from src.api.middleware.rate_limiter import RateLimitMiddleware
from src.api.routes import feedback, health, search
from src.config import get_settings
from src.monitoring.tracing import configure_tracing, StructuredLogger
from src.storage.cache import init_cache, close_cache
from src.storage.database import init_db, close_db

logger = logging.getLogger(__name__)
settings = get_settings()
structured_log = StructuredLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — runs at startup and shutdown.

    Startup:
    1. Configure tracing (LangSmith)
    2. Initialize DB connection pool
    3. Initialize Redis cache
    4. Warm up: verify all dependencies are healthy

    Shutdown:
    1. Drain in-flight requests
    2. Close DB + cache connections
    """
    # ── Startup ────────────────────────────────────────────────────────
    logger.info("Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.environment)

    configure_tracing()

    await init_db()
    logger.info("Database pool ready")

    await init_cache()
    logger.info("Redis cache ready")

    logger.info("All systems ready — serving requests")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("Shutting down...")
    await close_db()
    await close_cache()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Therapize API",
    description="AI-powered therapist search engine for California",
    version=settings.app_version,
    lifespan=lifespan,
    # Don't expose internal errors in production
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url=None,
)

# ── Middleware ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else ["https://therapize.app"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitMiddleware)


@app.middleware("http")
async def add_request_id(request: Request, call_next) -> Response:
    """Add X-Request-ID header to every request for distributed tracing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.monotonic()
    response = await call_next(request)
    latency_ms = (time.monotonic() - start_time) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-MS"] = str(round(latency_ms, 2))
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler.
    Returns 500 with structured error — never leaks stack traces in production.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("Unhandled exception [request_id=%s]", request_id)

    error_detail = str(exc) if settings.environment == "development" else "Internal server error"
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": error_detail,
            "request_id": request_id,
        },
    )


# ── Routes ──────────────────────────────────────────────────────────────────
app.include_router(search.router, prefix="/api/v1", tags=["search"])
app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])
app.include_router(health.router, tags=["health"])

# ── Prometheus metrics endpoint ──────────────────────────────────────────────
# Mount at /metrics for Prometheus scraper
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }
