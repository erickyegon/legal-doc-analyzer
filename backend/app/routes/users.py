"""
User management routes for the Legal Intelligence Platform.

This module handles user CRUD operations, profile management,
and user administration endpoints.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from app.database import get_db
from app.models.user import User, UserRole
from app.routes.auth import get_current_active_user, get_password_hash

# Initialize router
router = APIRouter()


# Pydantic models
class UserResponse(BaseModel):
    """User response model."""
    id: int
    username: str
    email: str
    full_name: str
    role: str
    is_active: bool
    is_verified: bool
    created_at: str
    last_login: Optional[str]
    organization: Optional[str]
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """User update model."""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone_number: Optional[str] = None
    organization: Optional[str] = None
    bio: Optional[str] = None


class UserCreate(BaseModel):
    """User creation model for admin."""
    username: str
    email: EmailStr
    full_name: str
    password: str
    role: UserRole
    organization: Optional[str] = None


class UserRoleUpdate(BaseModel):
    """User role update model."""
    role: UserRole


# Helper functions
def check_admin_permission(current_user: User):
    """Check if current user has admin permissions."""
    if not current_user.has_permission(UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )


# User endpoints
@router.get("/", response_model=List[UserResponse])
async def get_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    role: Optional[UserRole] = None,
    is_active: Optional[bool] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get list of users (admin only).
    
    Args:
        skip (int): Number of users to skip
        limit (int): Maximum number of users to return
        role (UserRole, optional): Filter by user role
        is_active (bool, optional): Filter by active status
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        List[UserResponse]: List of users
        
    Raises:
        HTTPException: If user doesn't have admin permissions
    """
    check_admin_permission(current_user)
    
    query = db.query(User)
    
    if role:
        query = query.filter(User.role == role)
    if is_active is not None:
        query = query.filter(User.is_active == is_active)
    
    users = query.offset(skip).limit(limit).all()
    return [UserResponse.from_orm(user) for user in users]


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get user by ID.
    
    Args:
        user_id (int): User ID
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        UserResponse: User information
        
    Raises:
        HTTPException: If user not found or insufficient permissions
    """
    # Users can view their own profile, admins can view any profile
    if user_id != current_user.id and not current_user.has_permission(UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.from_orm(user)


@router.post("/", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new user (admin only).
    
    Args:
        user_data (UserCreate): User creation data
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        UserResponse: Created user information
        
    Raises:
        HTTPException: If user doesn't have admin permissions or user already exists
    """
    check_admin_permission(current_user)
    
    # Check if username already exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=hashed_password,
        role=user_data.role,
        organization=user_data.organization,
        is_active=True,
        is_verified=True  # Admin-created users are auto-verified
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse.from_orm(db_user)


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update user information.
    
    Args:
        user_id (int): User ID
        user_data (UserUpdate): User update data
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        UserResponse: Updated user information
        
    Raises:
        HTTPException: If user not found or insufficient permissions
    """
    # Users can update their own profile, admins can update any profile
    if user_id != current_user.id and not current_user.has_permission(UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update user fields
    update_data = user_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)
    
    db.commit()
    db.refresh(user)
    
    return UserResponse.from_orm(user)


@router.patch("/{user_id}/role", response_model=UserResponse)
async def update_user_role(
    user_id: int,
    role_data: UserRoleUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update user role (admin only).
    
    Args:
        user_id (int): User ID
        role_data (UserRoleUpdate): Role update data
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        UserResponse: Updated user information
        
    Raises:
        HTTPException: If user not found or insufficient permissions
    """
    check_admin_permission(current_user)
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.role = role_data.role
    db.commit()
    db.refresh(user)
    
    return UserResponse.from_orm(user)


@router.patch("/{user_id}/activate")
async def activate_user(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Activate user account (admin only).
    
    Args:
        user_id (int): User ID
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        dict: Success message
        
    Raises:
        HTTPException: If user not found or insufficient permissions
    """
    check_admin_permission(current_user)
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.is_active = True
    db.commit()
    
    return {"message": "User activated successfully"}


@router.patch("/{user_id}/deactivate")
async def deactivate_user(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Deactivate user account (admin only).
    
    Args:
        user_id (int): User ID
        current_user (User): Current authenticated user
        db (Session): Database session
        
    Returns:
        dict: Success message
        
    Raises:
        HTTPException: If user not found or insufficient permissions
    """
    check_admin_permission(current_user)
    
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.is_active = False
    db.commit()
    
    return {"message": "User deactivated successfully"}
