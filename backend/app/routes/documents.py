"""
Document management routes for the Legal Intelligence Platform.

This module handles document upload, processing, retrieval,
and management operations for legal documents.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import os
import shutil
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
import magic
from datetime import datetime

from app.config import settings
from app.database import get_db
from app.models.document import Document, DocumentType, DocumentStatus
from app.models.user import User, UserRole
from app.routes.auth import get_current_active_user
from app.services.document_processor import DocumentProcessor
from app.services.file_handler import FileHandler

# Initialize router
router = APIRouter()

# Initialize services
document_processor = DocumentProcessor()
file_handler = FileHandler()


# Pydantic models
class DocumentResponse(BaseModel):
    """Document response model."""
    id: int
    filename: str
    file_size: int
    file_type: str
    document_type: str
    status: str
    title: Optional[str]
    description: Optional[str]
    content_preview: Optional[str]
    page_count: Optional[int]
    word_count: Optional[int]
    upload_date: str
    processed_date: Optional[str]
    owner_id: int
    is_confidential: bool
    tags: List[str]
    analysis_count: int

    class Config:
        from_attributes = True


class DocumentUpdate(BaseModel):
    """Document update model."""
    title: Optional[str] = None
    description: Optional[str] = None
    document_type: Optional[DocumentType] = None
    is_confidential: Optional[bool] = None
    tags: Optional[List[str]] = None


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    message: str
    document_id: int
    filename: str
    status: str


# Helper functions
def validate_file_type(file: UploadFile) -> bool:
    """Validate if the uploaded file type is allowed."""
    if file.content_type not in settings.allowed_file_types:
        return False
    return True


def validate_file_size(file: UploadFile) -> bool:
    """Validate if the uploaded file size is within limits."""
    # Note: This is a basic check. For more accurate size checking,
    # you might want to read the file in chunks
    return True  # Implement actual size checking logic


async def save_uploaded_file(file: UploadFile, user_id: int) -> str:
    """
    Save uploaded file to disk and return file path.

    Args:
        file (UploadFile): Uploaded file
        user_id (int): ID of the user uploading the file

    Returns:
        str: Path to saved file
    """
    # Create upload directory if it doesn't exist
    upload_dir = os.path.join(settings.upload_dir, str(user_id))
    os.makedirs(upload_dir, exist_ok=True)

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(upload_dir, filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path


# Document endpoints
@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_type: Optional[DocumentType] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    is_confidential: bool = False,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Upload a legal document for processing.

    Args:
        file (UploadFile): Document file to upload
        document_type (DocumentType, optional): Type of document
        title (str, optional): Document title
        description (str, optional): Document description
        is_confidential (bool): Whether document is confidential
        current_user (User): Current authenticated user
        db (Session): Database session

    Returns:
        DocumentUploadResponse: Upload response with document ID

    Raises:
        HTTPException: If file validation fails
    """
    # Validate file type
    if not validate_file_type(file):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file.content_type} not allowed"
        )

    # Validate file size
    if not validate_file_size(file):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum limit of {settings.max_file_size} bytes"
        )

    try:
        # Save file to disk
        file_path = await save_uploaded_file(file, current_user.id)

        # Get file size
        file_size = os.path.getsize(file_path)

        # Create document record
        document = Document(
            filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            file_type=file.content_type,
            document_type=document_type or DocumentType.OTHER,
            status=DocumentStatus.UPLOADED,
            title=title,
            description=description,
            owner_id=current_user.id,
            is_confidential=is_confidential
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        # Start background processing
        # TODO: Implement background task for document processing

        return DocumentUploadResponse(
            message="Document uploaded successfully",
            document_id=document.id,
            filename=document.filename,
            status=document.status.value
        )

    except Exception as e:
        # Clean up file if database operation fails
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )


@router.get("/", response_model=List[DocumentResponse])
async def get_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    document_type: Optional[DocumentType] = None,
    status: Optional[DocumentStatus] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get list of user's documents.

    Args:
        skip (int): Number of documents to skip
        limit (int): Maximum number of documents to return
        document_type (DocumentType, optional): Filter by document type
        status (DocumentStatus, optional): Filter by document status
        current_user (User): Current authenticated user
        db (Session): Database session

    Returns:
        List[DocumentResponse]: List of documents
    """
    query = db.query(Document).filter(Document.owner_id == current_user.id)

    if document_type:
        query = query.filter(Document.document_type == document_type)
    if status:
        query = query.filter(Document.status == status)

    documents = query.order_by(Document.upload_date.desc()).offset(skip).limit(limit).all()

    return [
        DocumentResponse(
            **doc.to_dict(),
            tags=doc.tags.split(",") if doc.tags else []
        )
        for doc in documents
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get document by ID.

    Args:
        document_id (int): Document ID
        current_user (User): Current authenticated user
        db (Session): Database session

    Returns:
        DocumentResponse: Document information

    Raises:
        HTTPException: If document not found or access denied
    """
    document = db.query(Document).filter(Document.id == document_id).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Check if user owns the document or has admin permissions
    if document.owner_id != current_user.id and not current_user.has_permission(UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    return DocumentResponse(
        **document.to_dict(),
        tags=document.tags.split(",") if document.tags else []
    )


@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: int,
    document_data: DocumentUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update document information.

    Args:
        document_id (int): Document ID
        document_data (DocumentUpdate): Document update data
        current_user (User): Current authenticated user
        db (Session): Database session

    Returns:
        DocumentResponse: Updated document information

    Raises:
        HTTPException: If document not found or access denied
    """
    document = db.query(Document).filter(Document.id == document_id).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Check if user owns the document
    if document.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Update document fields
    update_data = document_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        if field == "tags" and value:
            setattr(document, field, ",".join(value))
        else:
            setattr(document, field, value)

    db.commit()
    db.refresh(document)

    return DocumentResponse(
        **document.to_dict(),
        tags=document.tags.split(",") if document.tags else []
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete document.

    Args:
        document_id (int): Document ID
        current_user (User): Current authenticated user
        db (Session): Database session

    Returns:
        dict: Success message

    Raises:
        HTTPException: If document not found or access denied
    """
    document = db.query(Document).filter(Document.id == document_id).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Check if user owns the document or has admin permissions
    if document.owner_id != current_user.id and not current_user.has_permission(UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Delete file from disk
    try:
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
    except Exception as e:
        # Log error but don't fail the deletion
        print(f"Failed to delete file {document.file_path}: {e}")

    # Delete document from database
    db.delete(document)
    db.commit()

    return {"message": "Document deleted successfully"}