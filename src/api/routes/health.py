"""Health check endpoints for load balancers and monitoring."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.storage.cache import search_cache
from src.storage.database import TherapistRepository

router = APIRouter()
_repository = TherapistRepository()


@router.get("/health")
async def health_check() -> JSONResponse:
    """
    Shallow health check — fast response for load balancer checks.
    Returns 200 if the service is up, regardless of dependency health.
    """
    return JSONResponse({"status": "ok"})


@router.get("/health/deep")
async def deep_health_check() -> JSONResponse:
    """
    Deep health check — verifies all dependencies are reachable.
    Use for monitoring alerts, not load balancer checks (too slow).
    """
    db_ok = await _repository.ping()
    cache_ok = await search_cache.ping()

    all_ok = db_ok and cache_ok
    status = "healthy" if all_ok else "degraded"
    status_code = 200 if all_ok else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": status,
            "dependencies": {
                "database": "ok" if db_ok else "error",
                "cache": "ok" if cache_ok else "error",
            },
        },
    )
