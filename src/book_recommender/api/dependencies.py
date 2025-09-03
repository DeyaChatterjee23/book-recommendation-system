"""API dependencies for the Book Recommender System."""

import logging
from typing import Optional
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from book_recommender.core.config import get_settings
from book_recommender.core.security import security_manager
from book_recommender.core.exceptions import RateLimitExceededError
from book_recommender.services.recommendation_service import recommendation_service
from book_recommender.services.cache_service import cache_service

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


# Rate limiting storage (in production, use Redis)
rate_limit_storage = {}


def get_settings_dependency():
    """Get application settings."""
    return get_settings()


def get_recommendation_service():
    """Get recommendation service instance."""
    return recommendation_service


def get_cache_service():
    """Get cache service instance."""
    return cache_service


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """Get current user from JWT token (optional)."""
    if not credentials:
        return None

    try:
        payload = security_manager.verify_token(credentials.credentials)
        return {"user_id": payload.get("sub"), **payload}
    except Exception:
        # Don't raise error for optional authentication
        return None


async def get_current_user_required(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Get current user from JWT token (required)."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = security_manager.verify_token(credentials.credentials)
        return {"user_id": payload.get("sub"), **payload}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def check_rate_limit(
    request: Request,
    settings = Depends(get_settings_dependency)
) -> None:
    """Check rate limiting for API requests."""

    # Get client identifier (IP address or user ID)
    client_id = request.client.host

    # Get user from token if available
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            payload = security_manager.verify_token(token)
            client_id = payload.get("sub", client_id)  # Use user ID if available
        except Exception:
            pass  # Continue with IP-based rate limiting

    current_time = datetime.utcnow()
    window_start = current_time - timedelta(minutes=1)

    # Clean old entries
    if client_id in rate_limit_storage:
        rate_limit_storage[client_id] = [
            timestamp for timestamp in rate_limit_storage[client_id]
            if timestamp > window_start
        ]
    else:
        rate_limit_storage[client_id] = []

    # Check rate limit
    if len(rate_limit_storage[client_id]) >= settings.rate_limit_requests:
        logger.warning(f"Rate limit exceeded for client: {client_id}")
        raise RateLimitExceededError("Rate limit exceeded. Please try again later.")

    # Add current request timestamp
    rate_limit_storage[client_id].append(current_time)


def validate_pagination(
    limit: int = 20,
    offset: int = 0
) -> dict:
    """Validate pagination parameters."""
    if limit <= 0 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Limit must be between 1 and 100"
        )

    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Offset must be non-negative"
        )

    return {"limit": limit, "offset": offset}


def validate_num_recommendations(num_recommendations: int = 10) -> int:
    """Validate number of recommendations parameter."""
    if num_recommendations <= 0 or num_recommendations > 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Number of recommendations must be between 1 and 50"
        )
    return num_recommendations


async def check_model_availability(
    model_type: str,
    rec_service = Depends(get_recommendation_service)
) -> None:
    """Check if a specific model is available."""
    model_status = rec_service.get_model_status()

    if model_type == "collaborative":
        if model_status["collaborative_filtering"]["status"] != "trained":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Collaborative filtering model is not available"
            )
    elif model_type == "content":
        if model_status["content_based"]["status"] != "trained":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Content-based model is not available"
            )
    elif model_type == "hybrid":
        if model_status["hybrid"]["status"] != "trained":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Hybrid model is not available"
            )


def get_client_info(request: Request) -> dict:
    """Get client information from request."""
    return {
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent", "Unknown"),
        "endpoint": str(request.url),
        "method": request.method
    }


async def log_api_request(
    request: Request,
    user: Optional[dict] = Depends(get_current_user_optional)
) -> None:
    """Log API request for monitoring and analytics."""
    client_info = get_client_info(request)

    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user.get("user_id") if user else None,
        "endpoint": client_info["endpoint"],
        "method": client_info["method"],
        "ip_address": client_info["ip_address"],
        "user_agent": client_info["user_agent"]
    }

    logger.info(f"API Request: {log_data}")


# Admin authentication dependency
async def get_admin_user(
    current_user: dict = Depends(get_current_user_required)
) -> dict:
    """Require admin privileges."""
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


# Health check dependencies
async def check_system_health() -> dict:
    """Check overall system health."""
    health_status = {
        "status": "healthy",
        "checks": {},
        "timestamp": datetime.utcnow().isoformat()
    }

    # Check recommendation service
    try:
        model_status = recommendation_service.get_model_status()
        health_status["checks"]["recommendation_service"] = {
            "status": "healthy",
            "models_ready": model_status["system"]["models_ready"],
            "total_models": model_status["system"]["total_models"]
        }
    except Exception as e:
        health_status["checks"]["recommendation_service"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"

    # Check cache service
    try:
        cache_health = await cache_service.health_check()
        health_status["checks"]["cache_service"] = cache_health
        if not cache_health["healthy"]:
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["cache_service"] = {
            "status": "unhealthy",
            "healthy": False,
            "error": str(e)
        }
        health_status["status"] = "degraded"

    return health_status


# Request validation helpers
def validate_user_id(user_id: str) -> str:
    """Validate user ID format."""
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="User ID cannot be empty"
        )

    if len(user_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="User ID too long"
        )

    return user_id.strip()


def validate_book_id(book_id: int) -> int:
    """Validate book ID."""
    if book_id < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Book ID must be non-negative"
        )

    if book_id > 1000000:  # Reasonable upper bound
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Book ID too large"
        )

    return book_id


def validate_search_query(query: str) -> str:
    """Validate search query."""
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Search query cannot be empty"
        )

    if len(query) > 1000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Search query too long"
        )

    return query.strip()


# Metrics and monitoring
request_counts = {}
response_times = {}


async def track_metrics(request: Request, call_next):
    """Track API metrics."""
    start_time = datetime.utcnow()

    # Track request count
    endpoint = str(request.url.path)
    if endpoint not in request_counts:
        request_counts[endpoint] = 0
    request_counts[endpoint] += 1

    # Process request
    response = await call_next(request)

    # Track response time
    end_time = datetime.utcnow()
    response_time = (end_time - start_time).total_seconds()

    if endpoint not in response_times:
        response_times[endpoint] = []
    response_times[endpoint].append(response_time)

    # Keep only recent response times (last 1000 requests)
    if len(response_times[endpoint]) > 1000:
        response_times[endpoint] = response_times[endpoint][-1000:]

    # Add response time header
    response.headers["X-Response-Time"] = f"{response_time:.3f}s"

    return response


def get_metrics() -> dict:
    """Get API metrics."""
    metrics = {
        "request_counts": request_counts.copy(),
        "average_response_times": {},
        "total_requests": sum(request_counts.values())
    }

    # Calculate average response times
    for endpoint, times in response_times.items():
        if times:
            metrics["average_response_times"][endpoint] = sum(times) / len(times)

    return metrics


# Error handling helpers
def create_error_response(error_code: str, message: str, details: dict = None) -> dict:
    """Create standardized error response."""
    return {
        "success": False,
        "error_code": error_code,
        "message": message,
        "details": details or {},
        "timestamp": datetime.utcnow().isoformat()
    }
