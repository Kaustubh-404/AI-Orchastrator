# src/utils/media_storage.py
"""
Utility functions for reliably storing media files to the data directory.
"""

import os
import base64
import logging
from pathlib import Path
from typing import Optional, Union, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("media_storage")

def ensure_data_directory() -> Path:
    """
    Ensure that the data directory exists in the project root.
    Returns the path to the data directory.
    """
    # Try multiple approaches to find the project root
    possible_roots = [
        os.getcwd(),  # Current working directory
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Go up from src/utils
        os.path.dirname(os.path.abspath(__file__ if '__file__' in globals() else 'run.py')),  # File location
    ]
    
    data_dir = None
    for root in possible_roots:
        # Try to find the data directory
        possible_data_dir = os.path.join(root, "data")
        if os.path.exists(possible_data_dir):
            data_dir = possible_data_dir
            logger.info(f"Found existing data directory at: {data_dir}")
            break
        
        # If 'run.py' exists in this directory, we've found the project root
        elif os.path.exists(os.path.join(root, "run.py")):
            data_dir = os.path.join(root, "data")
            try:
                os.makedirs(data_dir, exist_ok=True)
                logger.info(f"Created data directory at: {data_dir}")
                break
            except Exception as e:
                logger.error(f"Failed to create data directory at {data_dir}: {str(e)}")
    
    if not data_dir:
        # Last resort: create a data directory in the current working directory
        data_dir = os.path.join(os.getcwd(), "data")
        try:
            os.makedirs(data_dir, exist_ok=True)
            logger.info(f"Created fallback data directory at: {data_dir}")
        except Exception as e:
            logger.error(f"Failed to create fallback data directory: {str(e)}")
    
    return Path(data_dir)

def save_base64_media(base64_data: str, filename: str, media_type: str = None) -> Tuple[bool, str]:
    """
    Save base64-encoded media data to the data directory.
    
    Args:
        base64_data: The base64-encoded data string (may include data URI prefix)
        filename: The filename to save as
        media_type: Optional media type to determine extension if not in filename
        
    Returns:
        Tuple of (success, file_path or error_message)
    """
    # Ensure data directory exists
    data_dir = ensure_data_directory()
    
    # Handle data URI prefix
    if ',' in base64_data:
        media_type_part, base64_data = base64_data.split(',', 1)
        # Extract media type from data URI if not provided
        if not media_type and media_type_part.startswith('data:'):
            media_type = media_type_part.split(';')[0][5:]
    
    # Add appropriate extension based on media type if missing
    if '.' not in filename:
        ext = '.bin'  # Default extension
        if media_type:
            if media_type.startswith('image/'):
                ext = '.png' if 'png' in media_type else '.jpg'
            elif media_type.startswith('audio/'):
                ext = '.wav' if 'wav' in media_type else '.mp3'
            elif media_type.startswith('video/'):
                ext = '.mp4'
        filename = f"{filename}{ext}"
    
    # Create full path
    file_path = data_dir / filename
    
    try:
        # Decode base64 and save to file
        file_bytes = base64.b64decode(base64_data)
        with open(file_path, 'wb') as f:
            f.write(file_bytes)
        
        logger.info(f"Successfully saved media to: {file_path}")
        return True, str(file_path)
    
    except Exception as e:
        error_msg = f"Failed to save media to {file_path}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def save_media_response(response_data: dict, prefix: str = "output") -> dict:
    """
    Extract and save media from API response data.
    
    Args:
        response_data: The response data from the AI Orchestrator API
        prefix: Prefix for generated filenames
        
    Returns:
        Dictionary with information about saved media files
    """
    result = {
        "saved_files": [],
        "errors": []
    }
    
    # Handle nested response structure
    if isinstance(response_data, dict) and "response" in response_data:
        response = response_data["response"]
    else:
        response = response_data
    
    # Handle dictionary response with media data
    if isinstance(response, dict):
        # Check for image data
        if "image_data" in response:
            success, path_or_error = save_base64_media(
                response["image_data"], 
                f"{prefix}_image", 
                "image/png"
            )
            if success:
                result["saved_files"].append({"type": "image", "path": path_or_error})
            else:
                result["errors"].append({"type": "image", "error": path_or_error})
        
        # Check for audio data
        if "audio_data" in response:
            success, path_or_error = save_base64_media(
                response["audio_data"], 
                f"{prefix}_audio", 
                "audio/wav"
            )
            if success:
                result["saved_files"].append({"type": "audio", "path": path_or_error})
            else:
                result["errors"].append({"type": "audio", "error": path_or_error})
        
        # Check for video data
        if "video_data" in response:
            success, path_or_error = save_base64_media(
                response["video_data"], 
                f"{prefix}_video", 
                "video/mp4"
            )
            if success:
                result["saved_files"].append({"type": "video", "path": path_or_error})
            else:
                result["errors"].append({"type": "video", "error": path_or_error})
        
        # Save text content if present
        if "text_content" in response:
            try:
                text_path = ensure_data_directory() / f"{prefix}_text.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(response["text_content"])
                result["saved_files"].append({"type": "text", "path": str(text_path)})
            except Exception as e:
                result["errors"].append({"type": "text", "error": str(e)})
    
    return result