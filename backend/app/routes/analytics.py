"""
Analytics routes for the Legal Intelligence Platform.

This module provides analytics and reporting endpoints for
system usage, document processing, and user activity metrics.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from pydantic import BaseModel

from app.database import get_db
from app.models.user import User, UserRole
from app.models.document import Document, DocumentType, DocumentStatus
from app.models.analysis import Analysis, AnalysisType, AnalysisStatus
from app.models.agent import Agent, AgentType
from app.routes.auth import get_current_active_user

# Initialize router
router = APIRouter()


# Pydantic models for analytics responses
class SystemStats(BaseModel):
    """System-wide statistics."""
    total_users: int
    active_users: int
    total_documents: int
    processed_documents: int
    total_analyses: int
    completed_analyses: int
    total_agents: int
    active_agents: int


class UserActivityStats(BaseModel):
    """User activity statistics."""
    user_id: int
    username: str
    documents_uploaded: int
    analyses_requested: int
    last_activity: Optional[str]


class DocumentStats(BaseModel):
    """Document processing statistics."""
    document_type: str
    total_count: int
    processed_count: int
    failed_count: int
    average_processing_time: Optional[float]


class AnalysisStats(BaseModel):
    """Analysis statistics."""
    analysis_type: str
    total_count: int
    completed_count: int
    failed_count: int
    average_processing_time: Optional[float]
    average_confidence_score: Optional[float]


class AgentPerformanceStats(BaseModel):
    """Agent performance statistics."""
    agent_id: int
    agent_name: str
    agent_type: str
    usage_count: int
    success_rate: float
    average_processing_time: float


class TimeSeriesData(BaseModel):
    """Time series data point."""
    date: str
    value: int


# Helper functions
def check_analytics_permission(current_user: User):
    """Check if user has permission to view analytics."""
    if not current_user.has_permission(UserRole.LAWYER):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to view analytics"
        )


def get_date_range(days: int) -> tuple:
    """Get date range for the specified number of days."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date


# Analytics endpoints
@router.get("/system-stats", response_model=SystemStats)
async def get_system_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get system-wide statistics.
    
    Args:
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        SystemStats: System statistics
        
    Raises:
        HTTPException: If user doesn't have sufficient permissions
    """
    check_analytics_permission(current_user)
    
    # User statistics
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    
    # Document statistics
    total_documents = db.query(Document).count()
    processed_documents = db.query(Document).filter(
        Document.status == DocumentStatus.PROCESSED
    ).count()
    
    # Analysis statistics
    total_analyses = db.query(Analysis).count()
    completed_analyses = db.query(Analysis).filter(
        Analysis.status == AnalysisStatus.COMPLETED
    ).count()
    
    # Agent statistics
    total_agents = db.query(Agent).count()
    active_agents = db.query(Agent).filter(Agent.is_available()).count()
    
    return SystemStats(
        total_users=total_users,
        active_users=active_users,
        total_documents=total_documents,
        processed_documents=processed_documents,
        total_analyses=total_analyses,
        completed_analyses=completed_analyses,
        total_agents=total_agents,
        active_agents=active_agents
    )


@router.get("/user-activity", response_model=List[UserActivityStats])
async def get_user_activity_stats(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(50, ge=1, le=1000),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get user activity statistics.
    
    Args:
        days (int): Number of days to look back
        limit (int): Maximum number of users to return
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        List[UserActivityStats]: User activity statistics
        
    Raises:
        HTTPException: If user doesn't have sufficient permissions
    """
    check_analytics_permission(current_user)
    
    start_date, end_date = get_date_range(days)
    
    # Query user activity
    user_stats = db.query(
        User.id,
        User.username,
        func.count(Document.id).label('documents_uploaded'),
        func.count(Analysis.id).label('analyses_requested'),
        func.max(User.last_login).label('last_activity')
    ).outerjoin(Document, User.id == Document.owner_id)\
     .outerjoin(Analysis, User.id == Analysis.created_by_id)\
     .filter(
         and_(
             Document.upload_date >= start_date,
             Document.upload_date <= end_date
         ) if Document.upload_date else True
     ).group_by(User.id, User.username)\
      .order_by(func.count(Document.id).desc())\
      .limit(limit).all()
    
    return [
        UserActivityStats(
            user_id=stat.id,
            username=stat.username,
            documents_uploaded=stat.documents_uploaded,
            analyses_requested=stat.analyses_requested,
            last_activity=stat.last_activity.isoformat() if stat.last_activity else None
        )
        for stat in user_stats
    ]


@router.get("/document-stats", response_model=List[DocumentStats])
async def get_document_stats(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get document processing statistics by type.
    
    Args:
        days (int): Number of days to look back
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        List[DocumentStats]: Document statistics by type
        
    Raises:
        HTTPException: If user doesn't have sufficient permissions
    """
    check_analytics_permission(current_user)
    
    start_date, end_date = get_date_range(days)
    
    # Query document statistics by type
    doc_stats = db.query(
        Document.document_type,
        func.count(Document.id).label('total_count'),
        func.sum(
            func.case(
                (Document.status == DocumentStatus.PROCESSED, 1),
                else_=0
            )
        ).label('processed_count'),
        func.sum(
            func.case(
                (Document.status == DocumentStatus.FAILED, 1),
                else_=0
            )
        ).label('failed_count')
    ).filter(
        and_(
            Document.upload_date >= start_date,
            Document.upload_date <= end_date
        )
    ).group_by(Document.document_type).all()
    
    return [
        DocumentStats(
            document_type=stat.document_type.value,
            total_count=stat.total_count,
            processed_count=stat.processed_count or 0,
            failed_count=stat.failed_count or 0,
            average_processing_time=None  # TODO: Calculate from processing logs
        )
        for stat in doc_stats
    ]


@router.get("/analysis-stats", response_model=List[AnalysisStats])
async def get_analysis_stats(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get analysis statistics by type.
    
    Args:
        days (int): Number of days to look back
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        List[AnalysisStats]: Analysis statistics by type
        
    Raises:
        HTTPException: If user doesn't have sufficient permissions
    """
    check_analytics_permission(current_user)
    
    start_date, end_date = get_date_range(days)
    
    # Query analysis statistics by type
    analysis_stats = db.query(
        Analysis.analysis_type,
        func.count(Analysis.id).label('total_count'),
        func.sum(
            func.case(
                (Analysis.status == AnalysisStatus.COMPLETED, 1),
                else_=0
            )
        ).label('completed_count'),
        func.sum(
            func.case(
                (Analysis.status == AnalysisStatus.FAILED, 1),
                else_=0
            )
        ).label('failed_count'),
        func.avg(Analysis.processing_time).label('avg_processing_time'),
        func.avg(Analysis.confidence_score).label('avg_confidence_score')
    ).filter(
        and_(
            Analysis.created_at >= start_date,
            Analysis.created_at <= end_date
        )
    ).group_by(Analysis.analysis_type).all()
    
    return [
        AnalysisStats(
            analysis_type=stat.analysis_type.value,
            total_count=stat.total_count,
            completed_count=stat.completed_count or 0,
            failed_count=stat.failed_count or 0,
            average_processing_time=float(stat.avg_processing_time) if stat.avg_processing_time else None,
            average_confidence_score=float(stat.avg_confidence_score) if stat.avg_confidence_score else None
        )
        for stat in analysis_stats
    ]


@router.get("/agent-performance", response_model=List[AgentPerformanceStats])
async def get_agent_performance_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get agent performance statistics.
    
    Args:
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        List[AgentPerformanceStats]: Agent performance statistics
        
    Raises:
        HTTPException: If user doesn't have sufficient permissions
    """
    check_analytics_permission(current_user)
    
    agents = db.query(Agent).all()
    
    return [
        AgentPerformanceStats(
            agent_id=agent.id,
            agent_name=agent.name,
            agent_type=agent.agent_type.value,
            usage_count=agent.usage_count,
            success_rate=agent.success_rate,
            average_processing_time=agent.average_processing_time
        )
        for agent in agents
    ]


@router.get("/documents-over-time", response_model=List[TimeSeriesData])
async def get_documents_over_time(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get document upload trends over time.
    
    Args:
        days (int): Number of days to look back
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        List[TimeSeriesData]: Document upload trends
        
    Raises:
        HTTPException: If user doesn't have sufficient permissions
    """
    check_analytics_permission(current_user)
    
    start_date, end_date = get_date_range(days)
    
    # Query documents by date
    daily_docs = db.query(
        func.date(Document.upload_date).label('date'),
        func.count(Document.id).label('count')
    ).filter(
        and_(
            Document.upload_date >= start_date,
            Document.upload_date <= end_date
        )
    ).group_by(func.date(Document.upload_date))\
     .order_by(func.date(Document.upload_date)).all()
    
    return [
        TimeSeriesData(
            date=doc.date.isoformat(),
            value=doc.count
        )
        for doc in daily_docs
    ]
