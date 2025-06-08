"""
File handling service for the Legal Intelligence Platform.

This module provides utilities for file operations, validation,
and secure file management.

Author: Legal Intelligence Platform Team
Version: 1.0.0
"""

import os
import shutil
import hashlib
import mimetypes
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class FileHandler:
    """
    Service class for handling file operations.
    
    This service provides secure file upload, storage, validation,
    and management capabilities for the platform.
    """
    
    def __init__(self):
        """Initialize the file handler."""
        self.upload_dir = settings.upload_dir
        self.max_file_size = settings.max_file_size
        self.allowed_types = settings.allowed_file_types
        
        # Ensure upload directory exists
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def validate_file_type(self, filename: str, content_type: str) -> bool:
        """
        Validate if the file type is allowed.
        
        Args:
            filename (str): Name of the file
            content_type (str): MIME type of the file
            
        Returns:
            bool: True if file type is allowed
        """
        # Check MIME type
        if content_type not in self.allowed_types:
            logger.warning(f"Disallowed MIME type: {content_type}")
            return False
        
        # Check file extension
        _, ext = os.path.splitext(filename.lower())
        allowed_extensions = {
            'application/pdf': ['.pdf'],
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
            'application/msword': ['.doc'],
            'text/plain': ['.txt']
        }
        
        if content_type in allowed_extensions:
            if ext not in allowed_extensions[content_type]:
                logger.warning(f"Extension {ext} doesn't match MIME type {content_type}")
                return False
        
        return True
    
    def validate_file_size(self, file_size: int) -> bool:
        """
        Validate if the file size is within limits.
        
        Args:
            file_size (int): Size of the file in bytes
            
        Returns:
            bool: True if file size is acceptable
        """
        if file_size > self.max_file_size:
            logger.warning(f"File size {file_size} exceeds limit {self.max_file_size}")
            return False
        
        if file_size == 0:
            logger.warning("Empty file not allowed")
            return False
        
        return True
    
    def generate_secure_filename(self, original_filename: str, user_id: int) -> str:
        """
        Generate a secure filename for storage.
        
        Args:
            original_filename (str): Original filename
            user_id (int): ID of the user uploading the file
            
        Returns:
            str: Secure filename
        """
        # Get file extension
        name, ext = os.path.splitext(original_filename)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create hash of original filename for uniqueness
        filename_hash = hashlib.md5(original_filename.encode()).hexdigest()[:8]
        
        # Combine components
        secure_filename = f"{user_id}_{timestamp}_{filename_hash}{ext}"
        
        return secure_filename
    
    def get_user_upload_dir(self, user_id: int) -> str:
        """
        Get the upload directory for a specific user.
        
        Args:
            user_id (int): User ID
            
        Returns:
            str: Path to user's upload directory
        """
        user_dir = os.path.join(self.upload_dir, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
    
    def save_file(self, file_content: bytes, filename: str, user_id: int) -> str:
        """
        Save file content to disk.
        
        Args:
            file_content (bytes): File content
            filename (str): Original filename
            user_id (int): User ID
            
        Returns:
            str: Path to saved file
            
        Raises:
            Exception: If file saving fails
        """
        try:
            # Generate secure filename
            secure_filename = self.generate_secure_filename(filename, user_id)
            
            # Get user directory
            user_dir = self.get_user_upload_dir(user_id)
            
            # Full file path
            file_path = os.path.join(user_dir, secure_filename)
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"File saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save file {filename}: {e}")
            raise
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from disk.
        
        Args:
            file_path (str): Path to file to delete
            
        Returns:
            bool: True if file was deleted successfully
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File deleted: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            dict: File information, or None if file doesn't exist
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            stat = os.stat(file_path)
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            
            return {
                'path': file_path,
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'mime_type': mime_type,
                'extension': os.path.splitext(file_path)[1].lower()
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return None
    
    def calculate_file_hash(self, file_path: str) -> Optional[str]:
        """
        Calculate SHA-256 hash of a file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            str: SHA-256 hash of the file, or None if calculation fails
        """
        try:
            hash_sha256 = hashlib.sha256()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            
            return hash_sha256.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return None
    
    def cleanup_user_files(self, user_id: int, older_than_days: int = 30) -> int:
        """
        Clean up old files for a user.
        
        Args:
            user_id (int): User ID
            older_than_days (int): Delete files older than this many days
            
        Returns:
            int: Number of files deleted
        """
        try:
            user_dir = os.path.join(self.upload_dir, str(user_id))
            
            if not os.path.exists(user_dir):
                return 0
            
            deleted_count = 0
            cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
            
            for filename in os.listdir(user_dir):
                file_path = os.path.join(user_dir, filename)
                
                if os.path.isfile(file_path):
                    file_stat = os.stat(file_path)
                    
                    if file_stat.st_mtime < cutoff_time:
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                            logger.info(f"Cleaned up old file: {file_path}")
                        except Exception as e:
                            logger.error(f"Failed to delete old file {file_path}: {e}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup files for user {user_id}: {e}")
            return 0
    
    def get_user_storage_usage(self, user_id: int) -> Dict[str, Any]:
        """
        Get storage usage statistics for a user.
        
        Args:
            user_id (int): User ID
            
        Returns:
            dict: Storage usage information
        """
        try:
            user_dir = os.path.join(self.upload_dir, str(user_id))
            
            if not os.path.exists(user_dir):
                return {
                    'total_files': 0,
                    'total_size': 0,
                    'average_file_size': 0,
                    'largest_file': None,
                    'oldest_file': None,
                    'newest_file': None
                }
            
            files_info = []
            total_size = 0
            
            for filename in os.listdir(user_dir):
                file_path = os.path.join(user_dir, filename)
                
                if os.path.isfile(file_path):
                    file_stat = os.stat(file_path)
                    file_info = {
                        'name': filename,
                        'path': file_path,
                        'size': file_stat.st_size,
                        'created': file_stat.st_ctime,
                        'modified': file_stat.st_mtime
                    }
                    files_info.append(file_info)
                    total_size += file_stat.st_size
            
            if not files_info:
                return {
                    'total_files': 0,
                    'total_size': 0,
                    'average_file_size': 0,
                    'largest_file': None,
                    'oldest_file': None,
                    'newest_file': None
                }
            
            # Calculate statistics
            largest_file = max(files_info, key=lambda x: x['size'])
            oldest_file = min(files_info, key=lambda x: x['created'])
            newest_file = max(files_info, key=lambda x: x['created'])
            
            return {
                'total_files': len(files_info),
                'total_size': total_size,
                'average_file_size': total_size // len(files_info),
                'largest_file': {
                    'name': largest_file['name'],
                    'size': largest_file['size']
                },
                'oldest_file': {
                    'name': oldest_file['name'],
                    'created': datetime.fromtimestamp(oldest_file['created'])
                },
                'newest_file': {
                    'name': newest_file['name'],
                    'created': datetime.fromtimestamp(newest_file['created'])
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage usage for user {user_id}: {e}")
            return {
                'total_files': 0,
                'total_size': 0,
                'average_file_size': 0,
                'largest_file': None,
                'oldest_file': None,
                'newest_file': None
            }
