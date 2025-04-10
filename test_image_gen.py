#!/usr/bin/env python
# test_image_gen.py - Standalone script to test image generation and storage

import os
import sys
import json
import time
import base64
import argparse
import requests
from pathlib import Path

def ensure_data_directory():
    """Create and return the data directory path."""
    # Try to find the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_roots = [
        script_dir,  # Current script directory
        os.path.dirname(script_dir) if os.path.basename(script_dir) in ('scripts', 'tests') else script_dir,
        os.getcwd(),  # Current working directory
    ]
    
    for root in possible_roots:
        # Check if this looks like the project root
        if os.path.exists(os.path.join(root, 'run.py')):
            data_dir = os.path.join(root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            print(f"Using data directory: {data_dir}")
            return data_dir
    
    # Fallback to creating a data directory in the script location
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    print(f"Using fallback data directory: {data_dir}")
    return data_dir

def save_base64_to_file(base64_data, output_path):
    """Save base64 data to a file."""
    # Remove data URI prefix if present
    if ',' in base64_data:
        base64_data = base64_data.split(',', 1)[1]
    
    with open(output_path, 'wb') as f:
        f.write(base64.b64decode(base64_data))
    
    print(f"Saved file to: {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
    return output_path

def make_request(prompt, base_url="http://localhost:8000"):
    """Make a request to generate an image and save it to the data directory."""
    data_dir = ensure_data_directory()
    timestamp = int(time.time())
    
    print(f"Making request with prompt: {prompt}")
    
    # Create the request
    try:
        response = requests.post(
            f"{base_url}/api/v1/requests",
            json={"content": prompt},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        request_data = response.json()
        request_id = request_data.get("request_id")
        
        if not request_id:
            print("Error: No request ID returned")
            return False
        
        print(f"Request created with ID: {request_id}")
    except Exception as e:
        print(f"Error creating request: {str(e)}")
        return False
    
    # Poll for completion
    max_attempts = 60
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        try:
            response = requests.get(f"{base_url}/api/v1/requests/{request_id}")
            response.raise_for_status()
            status_data = response.json()
            
            print(f"Status: {status_data.get('status')} (Attempt {attempt}/{max_attempts})")
            
            if status_data.get("status") == "completed":
                print("Request completed!")
                break
            elif status_data.get("status") == "failed":
                print(f"Request failed: {status_data.get('error')}")
                return False
            
            time.sleep(2)
        except Exception as e:
            print(f"Error checking status: {str(e)}")
            time.sleep(2)
    
    if attempt >= max_attempts:
        print("Timed out waiting for completion")
        return False
    
    # Save the response for debugging
    debug_path = os.path.join(data_dir, f"response_{request_id}.json")
    with open(debug_path, 'w') as f:
        json.dump(status_data, f, indent=2)
    print(f"Saved response data to {debug_path}")
    
    # Try to get the image from status data
    response_data = status_data.get("response", {})
    output_path = os.path.join(data_dir, f"image_{request_id}.png")
    
    # Case 1: response is a dictionary with image_data
    if isinstance(response_data, dict) and "image_data" in response_data:
        print("Found image_data in response dictionary")
        save_base64_to_file(response_data["image_data"], output_path)
        return output_path
    
    # Case 2: Try getting the image directly from the media endpoint
    try:
        print("Trying to get image from media endpoint...")
        response = requests.get(f"{base_url}/api/v1/media/{request_id}")
        
        if response.status_code == 200 and response.headers.get("Content-Type", "").startswith("image/"):
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Saved image from media endpoint to {output_path}")
            print(f"File size: {os.path.getsize(output_path)} bytes")
            return output_path
        else:
            print(f"Media endpoint returned non-image response: {response.status_code}")
    except Exception as e:
        print(f"Error getting image from media endpoint: {str(e)}")
    
    print("Failed to get image using any method")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test image generation")
    parser.add_argument("prompt", nargs="?", default="A beautiful sunset over mountains with pine trees",
                      help="Prompt for image generation")
    parser.add_argument("--url", default="http://localhost:8000", 
                      help="Base URL for the API server")
    args = parser.parse_args()
    
    output_path = make_request(args.prompt, args.url)
    
    if output_path:
        print("\nSUCCESS: Image generated and saved successfully!")
        # Try to open the image with the default viewer
        try:
            if sys.platform == 'win32':
                os.startfile(output_path)
            elif sys.platform == 'darwin':  # macOS
                os.system(f'open "{output_path}"')
            else:  # Linux
                os.system(f'xdg-open "{output_path}"')
        except Exception as e:
            print(f"Could not open the image automatically: {str(e)}")
    else:
        print("\nFAILURE: Could not generate or save the image.")
        sys.exit(1)