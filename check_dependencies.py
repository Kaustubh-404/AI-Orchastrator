#!/usr/bin/env python
# check_dependencies.py - Validate environment and dependencies for AI Orchestrator

import os
import sys
import subprocess
import importlib
import platform
import json
from pathlib import Path

def print_header(text):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)

def print_status(name, status, details=None):
    """Print a status line."""
    status_str = "[ OK ]" if status else "[FAIL]"
    status_color = "\033[92m" if status else "\033[91m"  # Green or Red
    reset_color = "\033[0m"
    
    print(f"{status_color}{status_str}{reset_color} {name}")
    if details and not status:
        print(f"       → {details}")

def check_directory_writeable(path):
    """Check if a directory is writeable."""
    if not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            return False, f"Cannot create directory: {str(e)}"
    
    try:
        test_file = os.path.join(path, ".write_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        return True, None
    except Exception as e:
        return False, f"Cannot write to directory: {str(e)}"

def check_python_packages():
    """Check if required Python packages are installed."""
    required_packages = [
        "fastapi", "uvicorn", "pydantic", "dotenv", "networkx", 
        "torch", "transformers", "diffusers", "pillow"
    ]
    
    results = []
    for package in required_packages:
        try:
            # Try to import the package
            if package == "dotenv":
                module = importlib.import_module("python-dotenv")
            else:
                module = importlib.import_module(package)
            
            # Get version if available
            try:
                version = module.__version__
            except AttributeError:
                version = "installed"
            
            results.append((package, True, version))
        except ImportError:
            results.append((package, False, "Not installed"))
    
    return results

def check_external_dependencies():
    """Check if external dependencies like ffmpeg are installed."""
    dependencies = [
        ("ffmpeg", ["ffmpeg", "-version"], "Required for video processing"),
    ]
    
    results = []
    for name, command, description in dependencies:
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
            version = output.split('\n')[0] if output else "Installed"
            results.append((name, True, version))
        except (subprocess.SubprocessError, FileNotFoundError):
            results.append((name, False, description))
    
    return results

def check_env_variables():
    """Check if required environment variables are set."""
    env_vars = [
        ("GROQ_API_KEY", "API key for Groq LLM service"),
        ("PORT", "Port for the FastAPI server (default: 8000)"),
        ("DEBUG", "Debug mode flag (default: false)"),
        ("LIGHTWEIGHT_MODEL", "Use lightweight models (default: false)"),
    ]
    
    results = []
    for var_name, description in env_vars:
        value = os.environ.get(var_name)
        if var_name == "GROQ_API_KEY":
            # Special handling for API key - don't show actual value
            if value:
                masked_value = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                results.append((var_name, True, f"Set to {masked_value}"))
            else:
                results.append((var_name, False, description))
        else:
            if value:
                results.append((var_name, True, f"Set to {value}"))
            else:
                is_required = var_name == "GROQ_API_KEY"
                results.append((var_name, not is_required, f"Not set - {description}"))
    
    return results

def check_data_directory():
    """Check if the data directory exists and is writeable."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_roots = [
        current_dir,
        os.path.dirname(current_dir),
        os.getcwd(),
    ]
    
    # Try to find the project root
    project_root = None
    for path in possible_roots:
        if os.path.exists(os.path.join(path, "run.py")):
            project_root = path
            break
    
    if not project_root:
        return False, "Project root not found"
    
    data_dir = os.path.join(project_root, "data")
    writable, error = check_directory_writeable(data_dir)
    
    if writable:
        return True, data_dir
    else:
        return False, f"Data directory issue: {error}"

def check_gpu():
    """Check if GPU is available for PyTorch."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if device_count > 0 else 0
            return True, f"CUDA available: {device_count} device(s), {device_name}, {memory:.2f} GB"
        else:
            return False, "CUDA not available"
    except (ImportError, Exception) as e:
        return False, f"Error checking GPU: {str(e)}"

def check_config_file():
    """Check if .env file exists and contains required variables."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, ".env"),
        os.path.join(os.path.dirname(current_dir), ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Check content
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    has_groq_key = "GROQ_API_KEY" in content
                    return True, f"Found at {path}" + (" (contains API key)" if has_groq_key else " (missing API key)")
            except Exception as e:
                return False, f"Error reading {path}: {str(e)}"
    
    return False, "Not found"

def main():
    """Run all checks and display results."""
    print_header("AI Orchestrator Environment Check")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Current Directory: {os.getcwd()}")
    
    # Check data directory
    print_header("Data Directory")
    data_status, data_details = check_data_directory()
    print_status("Data Directory", data_status, data_details)
    
    # Check environment variables
    print_header("Environment Variables")
    env_results = check_env_variables()
    for name, status, details in env_results:
        print_status(name, status, details)
    
    # Check config file
    print_header("Configuration")
    config_status, config_details = check_config_file()
    print_status(".env Configuration File", config_status, config_details)
    
    # Check Python packages
    print_header("Python Packages")
    package_results = check_python_packages()
    for name, status, details in package_results:
        print_status(name, status, details)
    
    # Check external dependencies
    print_header("External Dependencies")
    ext_results = check_external_dependencies()
    for name, status, details in ext_results:
        print_status(name, status, details)
    
    # Check GPU
    print_header("GPU Support")
    gpu_status, gpu_details = check_gpu()
    print_status("GPU", gpu_status, gpu_details)
    
    # Summarize any issues
    print_header("Summary")
    
    issues = []
    if not data_status:
        issues.append(f"Data directory issue: {data_details}")
    
    for name, status, details in env_results:
        if not status:
            issues.append(f"Missing environment variable: {name}")
    
    for name, status, details in package_results:
        if not status:
            issues.append(f"Missing Python package: {name}")
    
    for name, status, details in ext_results:
        if not status:
            issues.append(f"Missing external dependency: {name}")
    
    if issues:
        print(f"Found {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\nRecommendations:")
        print("  1. Create a 'data' directory in the project root if missing")
        print("  2. Install missing Python packages with: pip install -r requirements.txt")
        print("  3. Install ffmpeg if needed for video processing")
        print("  4. Set up a .env file with your GROQ_API_KEY")
    else:
        print("No issues found. System ready to run AI Orchestrator!")

if __name__ == "__main__":
    main()