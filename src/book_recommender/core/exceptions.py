"""Custom exceptions and error handlers for the Book Recommender System."""

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class BookRecommenderException(Exception):
    """Base exception for Book Recommender System."""

    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class DataNotFoundError(BookRecommenderException):
    """Raised when requested data is not found."""
    pass


class ModelNotLoadedError(BookRecommenderException):
    """Raised when a required model is not loaded."""
    pass


class ValidationError(BookRecommenderException):
    """Raised when data validation fails."""
    pass


class ExternalServiceError(BookRecommenderException):
    """Raised when external service call fails."""
    pass


class RateLimitExceededError(BookRecommenderException):
    """Raised when rate limit is exceeded."""
    pass


def create_error_response(
    status_code: int,
    message: str,
    error_code: str = None,
    details: Dict[str, Any] = None
) -> JSONResponse:
    """Create standardized error response."""

    error_response = {
        "error": {
            "message": message,
            "code": error_code or "GENERIC_ERROR",
            "status_code": status_code
        }
    }

    if details:
        error_response["error"]["details"] = details

    return JSONResponse(
        status_code=status_code,
        content=error_response
    )


async def book_recommender_exception_handler(
    request: Request, 
    exc: BookRecommenderException
) -> JSONResponse:
    """Handle custom BookRecommender exceptions."""

    logger.error(f"BookRecommender exception: {exc.message}", exc_info=True)

    if isinstance(exc, DataNotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, ModelNotLoadedError):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, ValidationError):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    elif isinstance(exc, ExternalServiceError):
        status_code = status.HTTP_502_BAD_GATEWAY
    elif isinstance(exc, RateLimitExceededError):
        status_code = status.HTTP_429_TOO_MANY_REQUESTS
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    return create_error_response(
        status_code=status_code,
        message=exc.message,
        error_code=exc.error_code
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""

    logger.warning(f"HTTP exception: {exc.detail}")

    return create_error_response(
        status_code=exc.status_code,
        message=exc.detail,
        error_code="HTTP_ERROR"
    )


async def validation_exception_handler(
    request: Request, 
    exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation exceptions."""

    logger.warning(f"Validation error: {exc.errors()}")

    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        message="Request validation failed",
        error_code="VALIDATION_ERROR",
        details={"validation_errors": exc.errors()}
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""

    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)

    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message="An unexpected error occurred",
        error_code="INTERNAL_ERROR"
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup all exception handlers for the FastAPI app."""

    app.add_exception_handler(BookRecommenderException, book_recommender_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
