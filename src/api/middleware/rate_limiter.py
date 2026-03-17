"""Sliding window rate limiter middleware."""
import logging

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.storage.cache import rate_limiter

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-IP rate limiting using Redis sliding window.
    Applies only to /api/ routes (not /health or /metrics).

    Response headers:
    - X-RateLimit-Remaining: requests remaining in current window
    - Retry-After: seconds until limit resets (on 429 response)
    """

    EXCLUDED_PATHS = {"/health", "/health/deep", "/metrics", "/"}

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip non-API paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        # Get client identifier (IP address, or auth token if authenticated)
        client_ip = request.client.host if request.client else "unknown"

        allowed, remaining = await rate_limiter.is_allowed(client_ip)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please wait before searching again.",
                    "retry_after_seconds": 60,
                },
                headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
