#!/usr/bin/env python3
"""
Main application entry point for the Book Recommender System.

This module initializes and runs the FastAPI application with proper configuration,
logging, and error handling.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from book_recommender.api.routes import router
from book_recommender.core.config import get_settings
from book_recommender.core.logger import setup_logging
from book_recommender.core.exceptions import setup_exception_handlers


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    # Load settings
    settings = get_settings()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        debug=settings.debug,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )

    # Setup exception handlers
    setup_exception_handlers(app)

    # Include routers
    app.include_router(router, prefix="/api/v1")

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "service": settings.app_name,
            "version": settings.app_version
        }

    logger.info(f"Application {settings.app_name} v{settings.app_version} initialized")
    return app


async def main():
    """Main application runner."""
    settings = get_settings()

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    app = create_app()

    # Run with uvicorn
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=1 if settings.reload else settings.workers,
        log_level=settings.log_level.lower(),
        access_log=True,
    )

    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Application stopped by user")
    except Exception as e:
        print(f"Application failed to start: {e}")
        sys.exit(1)
