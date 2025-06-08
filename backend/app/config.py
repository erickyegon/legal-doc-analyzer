"""
Configuration module for the Legal Intelligence Platform backend.

This module contains all configuration settings, environment variables,
and application constants used throughout the backend application.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import os
from typing import Optional
from pydantic import BaseSettings, validator
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    This class uses Pydantic BaseSettings to automatically load
    configuration from environment variables with type validation
    and default values.
    """

    # Application settings
    app_name: str = "Legal Intelligence Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # Database settings
    database_url: Optional[str] = None
    database_echo: bool = False

    # Security settings
    secret_key: str = "your-super-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours

    # EURI AI settings
    euri_api_key: Optional[str] = None
    euri_base_url: str = "https://api.euron.one/api/v1/euri/alpha/chat/completions"
    euri_model: str = "gpt-4.1-nano"
    euri_temperature: float = 0.7
    euri_max_tokens: int = 4000

    # File upload settings
    upload_dir: str = "uploads"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: list = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/plain"
    ]

    # CORS settings
    cors_origins: list = ["*"]  # Configure properly for production
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]

    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Redis settings (for caching and background tasks)
    redis_url: Optional[str] = None

    # Email settings (for notifications)
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True

    @validator("database_url", pre=True)
    def validate_database_url(cls, v):
        """Validate and set default database URL if not provided."""
        if v is None:
            return "sqlite:///./legal_intelligence.db"
        return v

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed_environments = ["development", "staging", "production"]
        if v not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v

    @validator("cors_origins", pre=True)
    def validate_cors_origins(cls, v):
        """Parse CORS origins from string if needed."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching.

    This function uses LRU cache to ensure settings are loaded only once
    and reused throughout the application lifecycle.

    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Application constants
class Constants:
    """Application-wide constants."""

    # Document processing
    MAX_DOCUMENT_PAGES = 1000
    MAX_ANALYSIS_RETRIES = 3
    ANALYSIS_TIMEOUT_SECONDS = 300  # 5 minutes

    # File handling
    CHUNK_SIZE = 8192  # For file reading
    PREVIEW_LENGTH = 500  # Characters for content preview

    # User roles and permissions
    ADMIN_PERMISSIONS = [
        "create_user", "delete_user", "manage_system",
        "view_all_documents", "delete_any_document"
    ]
    LAWYER_PERMISSIONS = [
        "upload_document", "analyze_document", "view_own_documents",
        "delete_own_document", "share_document"
    ]
    PARALEGAL_PERMISSIONS = [
        "upload_document", "analyze_document", "view_own_documents"
    ]
    VIEWER_PERMISSIONS = [
        "view_shared_documents"
    ]

    # Analysis types and their descriptions
    ANALYSIS_DESCRIPTIONS = {
        "summary": "Generate a comprehensive summary of the document",
        "risk_assessment": "Identify potential legal risks and liabilities",
        "clause_analysis": "Analyze contract clauses and terms",
        "compliance_check": "Check for regulatory compliance issues",
        "entity_extraction": "Extract legal entities and parties",
        "date_extraction": "Identify important dates and deadlines",
        "sentiment_analysis": "Analyze document tone and sentiment",
        "comparison": "Compare with other documents"
    }


# Export settings instance for easy importing
settings = get_settings()