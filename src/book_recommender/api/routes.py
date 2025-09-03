"""API routes for the Book Recommender System."""

import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status, BackgroundTasks

from book_recommender.api.schemas import *
from book_recommender.api.dependencies import (
    get_recommendation_service,
    get_cache_service,
    get_current_user_optional,
    get_current_user_required,
    get_admin_user,
    check_rate_limit,
    validate_num_recommendations,
    check_model_availability,
    log_api_request,
    check_system_health,
    validate_user_id,
    validate_book_id,
    validate_search_query,
    get_metrics,
    create_error_response
)
from book_recommender.services.recommendation_service import RecommendationService
from book_recommender.services.cache_service import CacheService
from book_recommender.core.exceptions import (
    ModelNotLoadedError,
    DataNotFoundError,
    ValidationError,
    RateLimitExceededError
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Health check endpoints
@router.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    health_data = await check_system_health()

    return HealthCheckResponse(
        status=health_data["status"],
        checks=health_data["checks"],
        version="1.0.0",
        message=f"System is {health_data['status']}"
    )


@router.get("/health/detailed", response_model=dict, tags=["Health"])
async def detailed_health_check(
    admin_user: dict = Depends(get_admin_user)
):
    """Detailed health check for administrators."""
    health_data = await check_system_health()

    # Add more detailed information
    health_data["metrics"] = get_metrics()
    health_data["admin_user"] = admin_user["user_id"]

    return health_data


# Model status endpoints
@router.get("/models/status", response_model=ModelStatusResponse, tags=["Models"])
async def get_model_status(
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    """Get status of all recommendation models."""
    try:
        status_data = rec_service.get_model_status()

        return ModelStatusResponse(
            collaborative_filtering=ModelInfo(**status_data["collaborative_filtering"]),
            content_based=ModelInfo(**status_data["content_based"]),
            hybrid=ModelInfo(**status_data["hybrid"]),
            cache_service=status_data["cache_service"],
            system=status_data["system"],
            message="Model status retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model status"
        )


# Recommendation endpoints
@router.get(
    "/recommendations/collaborative/{user_id}",
    response_model=CollaborativeRecommendationResponse,
    dependencies=[Depends(check_rate_limit), Depends(log_api_request)],
    tags=["Recommendations"]
)
async def get_collaborative_recommendations(
    user_id: str = Path(..., description="User ID for recommendations"),
    num_recommendations: int = Query(default=10, description="Number of recommendations"),
    rec_service: RecommendationService = Depends(get_recommendation_service),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Get collaborative filtering recommendations for a user."""

    # Validate inputs
    user_id = validate_user_id(user_id)
    num_recommendations = validate_num_recommendations(num_recommendations)

    # Check model availability
    await check_model_availability("collaborative", rec_service)

    try:
        result = await rec_service.get_collaborative_recommendations(user_id, num_recommendations)

        return CollaborativeRecommendationResponse(
            user_id=result["user_id"],
            model_type=result["model_type"],
            recommendations=[BookRecommendation(**rec) for rec in result["recommendations"]],
            total_count=result["total_count"],
            message=f"Generated {result['total_count']} collaborative recommendations"
        )

    except ModelNotLoadedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Collaborative recommendations failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )


@router.get(
    "/recommendations/content/{book_id}",
    response_model=ContentRecommendationResponse,
    dependencies=[Depends(check_rate_limit), Depends(log_api_request)],
    tags=["Recommendations"]
)
async def get_content_based_recommendations(
    book_id: int = Path(..., description="Book ID for recommendations"),
    num_recommendations: int = Query(default=10, description="Number of recommendations"),
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    """Get content-based recommendations for a book."""

    # Validate inputs
    book_id = validate_book_id(book_id)
    num_recommendations = validate_num_recommendations(num_recommendations)

    # Check model availability
    await check_model_availability("content", rec_service)

    try:
        result = await rec_service.get_content_based_recommendations(book_id, num_recommendations)

        return ContentRecommendationResponse(
            book_id=result["book_id"],
            book_details=BookDetails(**result["book_details"]),
            model_type=result["model_type"],
            recommendations=[BookRecommendation(**rec) for rec in result["recommendations"]],
            total_count=result["total_count"],
            message=f"Generated {result['total_count']} content-based recommendations"
        )

    except DataNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Content recommendations failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )


@router.post(
    "/recommendations/hybrid",
    response_model=HybridRecommendationResponse,
    dependencies=[Depends(check_rate_limit), Depends(log_api_request)],
    tags=["Recommendations"]
)
async def get_hybrid_recommendations(
    request: HybridRecommendationRequest,
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    """Get hybrid recommendations."""

    # Check model availability
    await check_model_availability("hybrid", rec_service)

    try:
        result = await rec_service.get_hybrid_recommendations(
            user_id=request.user_id,
            book_id=request.book_id,
            preferences=request.preferences.dict() if request.preferences else None,
            num_recommendations=request.num_recommendations
        )

        return HybridRecommendationResponse(
            model_type=result["model_type"],
            input_data=result["input_data"],
            recommendations=[BookRecommendation(**rec) for rec in result["recommendations"]],
            total_count=result["total_count"],
            message=f"Generated {result['total_count']} hybrid recommendations"
        )

    except ModelNotLoadedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Hybrid recommendations failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations"
        )


# Book-related endpoints
@router.get(
    "/books/search",
    response_model=BookSearchResponse,
    dependencies=[Depends(check_rate_limit), Depends(log_api_request)],
    tags=["Books"]
)
async def search_books(
    query: str = Query(..., description="Search query"),
    num_results: int = Query(default=20, description="Number of results"),
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    """Search for books by text query."""

    # Validate inputs
    query = validate_search_query(query)

    if num_results <= 0 or num_results > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Number of results must be between 1 and 100"
        )

    try:
        result = await rec_service.search_books(query, num_results)

        return BookSearchResponse(
            query=result["query"],
            results=[SearchResult(**res) for res in result["results"]],
            total_count=result["total_count"],
            message=f"Found {result['total_count']} books matching query"
        )

    except ModelNotLoadedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Book search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed"
        )


@router.get(
    "/books/{book_id}",
    response_model=BookDetailsResponse,
    dependencies=[Depends(check_rate_limit), Depends(log_api_request)],
    tags=["Books"]
)
async def get_book_details(
    book_id: int = Path(..., description="Book ID"),
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    """Get detailed information about a book."""

    # Validate input
    book_id = validate_book_id(book_id)

    try:
        book_details = await rec_service.get_book_details(book_id)

        return BookDetailsResponse(
            book=BookDetails(**book_details),
            message="Book details retrieved successfully"
        )

    except DataNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Get book details failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve book details"
        )


@router.post(
    "/books/similar",
    response_model=SimilarBooksResponse,
    dependencies=[Depends(check_rate_limit), Depends(log_api_request)],
    tags=["Books"]
)
async def get_similar_books(
    request: SimilarBooksRequest,
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    """Get books similar to a given book title."""

    try:
        result = await rec_service.get_similar_books(request.book_title, request.num_recommendations)

        return SimilarBooksResponse(
            book_title=result["book_title"],
            model_type=result["model_type"],
            recommendations=[BookRecommendation(**rec) for rec in result["recommendations"]],
            total_count=result["total_count"],
            message=f"Found {result['total_count']} similar books"
        )

    except DataNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Similar books search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to find similar books"
        )


@router.get(
    "/books/popular",
    response_model=PopularBooksResponse,
    dependencies=[Depends(check_rate_limit), Depends(log_api_request)],
    tags=["Books"]
)
async def get_popular_books(
    num_books: int = Query(default=20, description="Number of popular books"),
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    """Get popular books."""

    if num_books <= 0 or num_books > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Number of books must be between 1 and 100"
        )

    try:
        result = await rec_service.get_popular_books(num_books)

        return PopularBooksResponse(
            model_type=result["model_type"],
            books=[BookRecommendation(**book) for book in result["books"]],
            total_count=result["total_count"],
            message=f"Retrieved {result['total_count']} popular books"
        )

    except Exception as e:
        logger.error(f"Get popular books failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve popular books"
        )


# Cache management endpoints (Admin only)
@router.get(
    "/cache/info",
    response_model=dict,
    dependencies=[Depends(get_admin_user)],
    tags=["Cache"]
)
async def get_cache_info(
    cache_service: CacheService = Depends(get_cache_service)
):
    """Get cache service information."""
    try:
        cache_info = await cache_service.get_info()
        return {"success": True, "cache_info": cache_info}
    except Exception as e:
        logger.error(f"Get cache info failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache information"
        )


@router.delete(
    "/cache/clear",
    response_model=CacheOperationResponse,
    dependencies=[Depends(get_admin_user)],
    tags=["Cache"]
)
async def clear_cache(
    pattern: Optional[str] = Query(None, description="Pattern to match keys (optional)"),
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    """Clear cache entries."""
    try:
        result = await rec_service.clear_cache(pattern)

        return CacheOperationResponse(
            operation="clear",
            result=result,
            message=f"Cleared {result['cleared_keys']} cache entries"
        )
    except Exception as e:
        logger.error(f"Clear cache failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )


@router.post(
    "/cache/warmup",
    response_model=CacheWarmupResponse,
    dependencies=[Depends(get_admin_user)],
    tags=["Cache"]
)
async def warmup_cache(
    request: CacheWarmupRequest,
    background_tasks: BackgroundTasks,
    rec_service: RecommendationService = Depends(get_recommendation_service)
):
    """Warm up cache with sample requests."""

    async def run_warmup():
        try:
            start_time = datetime.utcnow()
            result = await rec_service.warmup_cache(request.sample_requests)
            end_time = datetime.utcnow()

            duration = (end_time - start_time).total_seconds()
            logger.info(f"Cache warmup completed in {duration:.2f} seconds: {result}")

        except Exception as e:
            logger.error(f"Cache warmup failed: {e}")

    # Run warmup in background
    background_tasks.add_task(run_warmup)

    return CacheWarmupResponse(
        total_requests=len(request.sample_requests),
        successful=0,  # Will be updated by background task
        failed=0,
        duration_seconds=0,
        message="Cache warmup started in background"
    )


# Batch operations
@router.post(
    "/recommendations/batch",
    response_model=BatchRecommendationResponse,
    dependencies=[Depends(get_admin_user)],  # Restrict to admin users
    tags=["Batch Operations"]
)
async def batch_recommendations(
    request: BatchRecommendationRequest
):
    """Process batch recommendation requests."""

    results = []
    errors = []
    successful = 0
    failed = 0

    for req in request.requests:
        try:
            if isinstance(req, CollaborativeRecommendationRequest):
                result = await get_collaborative_recommendations(
                    user_id=req.user_id,
                    num_recommendations=req.num_recommendations
                )
            elif isinstance(req, ContentRecommendationRequest):
                result = await get_content_based_recommendations(
                    book_id=req.book_id,
                    num_recommendations=req.num_recommendations
                )
            elif isinstance(req, HybridRecommendationRequest):
                result = await get_hybrid_recommendations(req)
            else:
                raise ValueError(f"Unsupported request type: {type(req)}")

            results.append(result)
            successful += 1

        except Exception as e:
            errors.append(str(e))
            failed += 1

    return BatchRecommendationResponse(
        results=results,
        successful=successful,
        failed=failed,
        errors=errors,
        message=f"Batch processing completed: {successful} successful, {failed} failed"
    )


# Metrics endpoint (Admin only)
@router.get(
    "/metrics",
    response_model=dict,
    dependencies=[Depends(get_admin_user)],
    tags=["Metrics"]
)
async def get_api_metrics():
    """Get API usage metrics."""
    try:
        metrics = get_metrics()
        return {
            "success": True,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Get metrics failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


# Rate limit info endpoint
@router.get(
    "/ratelimit/info",
    response_model=dict,
    tags=["Rate Limit"]
)
async def get_rate_limit_info(
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Get rate limit information for the current client."""
    from book_recommender.api.dependencies import rate_limit_storage
    from book_recommender.core.config import get_settings

    settings = get_settings()

    # Get client identifier
    client_id = current_user.get("user_id") if current_user else "anonymous"

    # Get current usage
    current_usage = len(rate_limit_storage.get(client_id, []))

    return {
        "rate_limit": settings.rate_limit_requests,
        "current_usage": current_usage,
        "remaining": max(0, settings.rate_limit_requests - current_usage),
        "window_minutes": 1,
        "client_type": "authenticated" if current_user else "anonymous"
    }


# Error handling
@router.exception_handler(RateLimitExceededError)
async def rate_limit_exception_handler(request, exc):
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=str(exc)
    )


# Root endpoint
@router.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "message": "Book Recommender System API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "status": "operational"
    }
