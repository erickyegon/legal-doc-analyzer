"""
Database configuration and initialization module.

This module sets up the database connection, session management, and base model
for the Legal Intelligence Platform. It supports both SQLite (development) and
PostgreSQL (production) databases.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import os
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")

# If no DATABASE_URL is provided, use SQLite for development
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./legal_intelligence.db"
    logger.info("Using SQLite database for development")
    
    # SQLite specific configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=True  # Set to False in production
    )
else:
    logger.info("Using PostgreSQL database for production")
    
    # PostgreSQL configuration for production
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False  # Disable SQL logging in production
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Metadata for database operations
metadata = MetaData()


def get_db():
    """
    Dependency function to get database session.
    
    This function creates a new database session for each request
    and ensures it's properly closed after use.
    
    Yields:
        Session: SQLAlchemy database session
        
    Example:
        @app.get("/users/")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Initialize the database by creating all tables.
    
    This function should be called when the application starts
    to ensure all database tables are created.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def check_db_connection():
    """
    Check if the database connection is working.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


# Export commonly used objects
__all__ = [
    "engine",
    "SessionLocal", 
    "Base",
    "get_db",
    "init_db",
    "check_db_connection",
    "metadata"
]
