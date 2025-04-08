import json
import time
import uuid
from typing import Dict, Any, List, Optional
import os
from datetime import datetime


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def get_env_var(name: str, default: Optional[str] = None) -> str:
    """
    Get environment variable or return default value.
    
    Args:
        name: Name of the environment variable
        default: Default value if not found
        
    Returns:
        Value of the environment variable or default
    """
    return os.environ.get(name, default)


def current_timestamp() -> float:
    """Get current timestamp."""
    return time.time()


def format_timestamp(timestamp: float) -> str:
    """
    Format a timestamp as ISO string.
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Formatted timestamp string
    """
    return datetime.fromtimestamp(timestamp).isoformat()


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON string.
    
    Args:
        json_str: JSON string to load
        default: Default value if loading fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to max_length characters.
    
    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def filter_dict(d: Dict, keys: List[str]) -> Dict:
    """
    Filter a dictionary to only include specific keys.
    
    Args:
        d: Dictionary to filter
        keys: Keys to include
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in d.items() if k in keys}


def merge_dicts(dict1: Dict, dict2: Dict, overwrite: bool = True) -> Dict:
    """
    Merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        overwrite: Whether to overwrite values in dict1 with values from dict2
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key not in result or overwrite:
            result[key] = value
    
    return result


def format_error(error: Exception) -> Dict[str, Any]:
    """
    Format an exception as a dictionary.
    
    Args:
        error: Exception to format
        
    Returns:
        Dictionary with error details
    """
    return {
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "timestamp": current_timestamp()
    }