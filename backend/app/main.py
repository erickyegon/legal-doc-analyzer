"""
Main FastAPI application module for the Legal Intelligence Platform.

This module initializes the FastAPI application, configures middleware,
sets up database connections, and includes all route handlers.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import application modules
from app.config import settings
from app.database import init_db, check_db_connection
from app.routes import auth, documents, agents, users, analytics
from app.models import *  # Import all models to ensure they're registered

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events for the FastAPI application.
    This includes database initialization and cleanup tasks.
    """
    # Startup
    logger.info("Starting Legal Intelligence Platform...")

    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")

        # Check database connection
        if not check_db_connection():
            logger.error("Database connection failed")
            raise HTTPException(status_code=500, detail="Database connection failed")

        logger.info("Application startup completed")

    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Legal Intelligence Platform...")


# Create FastAPI application instance
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A comprehensive legal document analysis platform powered by AI",
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan
)

# Security middleware
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.onrender.com", "localhost", "127.0.0.1"]
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled exceptions.

    Args:
        request: The request that caused the exception
        exc: The exception that was raised

    Returns:
        JSONResponse: Error response with appropriate status code
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    if settings.environment == "production":
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    else:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)}
        )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.

    Returns:
        dict: Application health status
    """
    try:
        db_status = check_db_connection()

        return {
            "status": "healthy" if db_status else "unhealthy",
            "version": settings.app_version,
            "environment": settings.environment,
            "database": "connected" if db_status else "disconnected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# Root endpoint
@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint providing basic application information.

    Returns:
        dict: Welcome message and application info
    """
    return {
        "message": "Welcome to the Legal Intelligence Platform!",
        "version": settings.app_version,
        "docs": "/docs" if settings.environment != "production" else "Contact administrator",
        "health": "/health"
    }


# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agents"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )