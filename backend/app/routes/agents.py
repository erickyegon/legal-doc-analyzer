"""
Agent management and analysis routes for the Legal Intelligence Platform.

This module handles AI agent operations, document analysis requests,
and agent performance monitoring.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.models.agent import Agent, AgentType, AgentStatus
from app.models.document import Document, DocumentStatus
from app.models.analysis import Analysis, AnalysisType, AnalysisStatus
from app.models.user import User, UserRole
from app.routes.auth import get_current_active_user
from app.services.analysis_service import AnalysisService

# Initialize router
router = APIRouter()

# Initialize services
analysis_service = AnalysisService()


# Pydantic models
class AgentResponse(BaseModel):
    """Agent response model."""
    id: int
    name: str
    agent_type: str
    status: str
    description: Optional[str]
    version: str
    model_name: str
    capabilities: Optional[dict]
    usage_count: int
    success_rate: float
    average_processing_time: float
    is_default: bool

    class Config:
        from_attributes = True


class AnalysisRequest(BaseModel):
    """Analysis request model."""
    document_id: int
    analysis_type: AnalysisType
    agent_id: Optional[int] = None
    title: Optional[str] = None


class AnalysisResponse(BaseModel):
    """Analysis response model."""
    id: int
    document_id: int
    analysis_type: str
    status: str
    title: Optional[str]
    summary: Optional[str]
    content: Optional[str]
    confidence_score: Optional[float]
    processing_time: Optional[float]
    created_at: str
    completed_at: Optional[str]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class AgentCreate(BaseModel):
    """Agent creation model."""
    name: str
    agent_type: AgentType
    description: Optional[str] = None
    model_name: str
    model_version: Optional[str] = None
    capabilities: Optional[dict] = None
    configuration: Optional[dict] = None
    max_tokens: int = 4000
    temperature: float = 0.7


class AgentUpdate(BaseModel):
    """Agent update model."""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[AgentStatus] = None
    capabilities: Optional[dict] = None
    configuration: Optional[dict] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


# Helper functions
def check_agent_admin_permission(current_user: User):
    """Check if user has permission to manage agents."""
    if not current_user.has_permission(UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to manage agents"
        )


# Agent endpoints
@router.get("/", response_model=List[AgentResponse])
async def get_agents(
    agent_type: Optional[AgentType] = None,
    status: Optional[AgentStatus] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get list of available agents.

    Args:
        agent_type (AgentType, optional): Filter by agent type
        status (AgentStatus, optional): Filter by agent status
        current_user (User): Current authenticated user
        db (Session): Database session

    Returns:
        List[AgentResponse]: List of agents
    """
    query = db.query(Agent)

    if agent_type:
        query = query.filter(Agent.agent_type == agent_type)
    if status:
        query = query.filter(Agent.status == status)

    agents = query.order_by(Agent.name).all()

    return [AgentResponse.from_orm(agent) for agent in agents]


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get agent by ID.

    Args:
        agent_id (int): Agent ID
        current_user (User): Current authenticated user
        db (Session): Database session

    Returns:
        AgentResponse: Agent information

    Raises:
        HTTPException: If agent not found
    """
    agent = db.query(Agent).filter(Agent.id == agent_id).first()

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )

    return AgentResponse.from_orm(agent)


@router.post("/", response_model=AgentResponse)
async def create_agent(
    agent_data: AgentCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new agent (admin only).

    Args:
        agent_data (AgentCreate): Agent creation data
        current_user (User): Current authenticated user
        db (Session): Database session

    Returns:
        AgentResponse: Created agent information

    Raises:
        HTTPException: If user doesn't have admin permissions
    """
    check_agent_admin_permission(current_user)

    # Check if agent name already exists
    if db.query(Agent).filter(Agent.name == agent_data.name).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent name already exists"
        )

    # Create new agent
    agent = Agent(
        name=agent_data.name,
        agent_type=agent_data.agent_type,
        description=agent_data.description,
        version="1.0.0",  # Default version
        model_name=agent_data.model_name,
        model_version=agent_data.model_version,
        capabilities=agent_data.capabilities,
        configuration=agent_data.configuration,
        max_tokens=agent_data.max_tokens,
        temperature=agent_data.temperature,
        status=AgentStatus.ACTIVE
    )

    db.add(agent)
    db.commit()
    db.refresh(agent)

    return AgentResponse.from_orm(agent)


@router.post("/analyze", response_model=AnalysisResponse)
async def request_analysis(
    analysis_request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Request document analysis by an AI agent.

    Args:
        analysis_request (AnalysisRequest): Analysis request data
        background_tasks (BackgroundTasks): FastAPI background tasks
        current_user (User): Current authenticated user
        db (Session): Database session

    Returns:
        AnalysisResponse: Analysis information

    Raises:
        HTTPException: If document not found or access denied
    """
    # Check if document exists and user has access
    document = db.query(Document).filter(Document.id == analysis_request.document_id).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    if document.owner_id != current_user.id and not current_user.has_permission(UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Check if document is processed
    if document.status != DocumentStatus.PROCESSED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document must be processed before analysis"
        )

    # Get agent (use default if not specified)
    agent = None
    if analysis_request.agent_id:
        agent = db.query(Agent).filter(Agent.id == analysis_request.agent_id).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
    else:
        # Find default agent for the analysis type
        agent = db.query(Agent).filter(
            Agent.agent_type == AgentType.GENERAL_LEGAL,
            Agent.status == AgentStatus.ACTIVE,
            Agent.is_default == True
        ).first()

        if not agent:
            # Fallback to any active agent
            agent = db.query(Agent).filter(
                Agent.status == AgentStatus.ACTIVE
            ).first()

    if not agent:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No agents available for analysis"
        )

    if not agent.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Selected agent is not available"
        )

    # Create analysis record
    analysis = Analysis(
        document_id=document.id,
        created_by_id=current_user.id,
        analysis_type=analysis_request.analysis_type,
        title=analysis_request.title,
        status=AnalysisStatus.PENDING,
        model_version=agent.model_version
    )

    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    # Start background analysis task
    background_tasks.add_task(
        analysis_service.process_analysis,
        analysis.id,
        agent.id,
        document.file_path
    )

    return AnalysisResponse.from_orm(analysis)


@router.get("/analyses", response_model=List[AnalysisResponse])
async def get_analyses(
    document_id: Optional[int] = None,
    analysis_type: Optional[AnalysisType] = None,
    status: Optional[AnalysisStatus] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get user's analyses.

    Args:
        document_id (int, optional): Filter by document ID
        analysis_type (AnalysisType, optional): Filter by analysis type
        status (AnalysisStatus, optional): Filter by analysis status
        current_user (User): Current authenticated user
        db (Session): Database session

    Returns:
        List[AnalysisResponse]: List of analyses
    """
    query = db.query(Analysis).filter(Analysis.created_by_id == current_user.id)

    if document_id:
        query = query.filter(Analysis.document_id == document_id)
    if analysis_type:
        query = query.filter(Analysis.analysis_type == analysis_type)
    if status:
        query = query.filter(Analysis.status == status)

    analyses = query.order_by(Analysis.created_at.desc()).all()

    return [AnalysisResponse.from_orm(analysis) for analysis in analyses]


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get analysis by ID.

    Args:
        analysis_id (int): Analysis ID
        current_user (User): Current authenticated user
        db (Session): Database session

    Returns:
        AnalysisResponse: Analysis information

    Raises:
        HTTPException: If analysis not found or access denied
    """
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()

    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )

    # Check if user owns the analysis or has admin permissions
    if analysis.created_by_id != current_user.id and not current_user.has_permission(UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    return AnalysisResponse.from_orm(analysis)