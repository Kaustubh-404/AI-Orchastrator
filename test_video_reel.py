#!/usr/bin/env python
# test_video_reel.py - Test video reel generation

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_roots = [
        script_dir,
        os.path.dirname(script_dir) if os.path.basename(script_dir) in ('scripts', 'tests') else script_dir,
        os.getcwd(),
    ]
    
    for root in possible_roots:
        if os.path.exists(os.path.join(root, 'run.py')):
            data_dir = os.path.join(root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            print(f"Using data directory: {data_dir}")
            return data_dir
    
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
    """Make a request to generate a video reel and save it to the data directory."""
    data_dir = ensure_data_directory()
    
    print(f"Making video reel request: {prompt}")
    
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
    
    # Poll until completion without timeout
    print("Waiting for processing to complete...")
    attempt = 0
    
    while True:
        attempt += 1
        try:
            response = requests.get(f"{base_url}/api/v1/requests/{request_id}")
            response.raise_for_status()
            status_data = response.json()
            
            print(f"Status: {status_data.get('status')} (Attempt {attempt})")
            
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
    
    # Save the response for debugging
    debug_path = os.path.join(data_dir, f"response_{request_id}.json")
    with open(debug_path, 'w') as f:
        json.dump(status_data, f, indent=2)
    print(f"Saved response data to {debug_path}")
    
    # Process the response
    response_data = status_data.get("response", {})
    results = []
    
    # Handle text content
    text_content = None
    if isinstance(response_data, dict) and "text_content" in response_data:
        text_content = response_data["text_content"]
    
    if text_content:
        text_path = os.path.join(data_dir, f"script_{request_id}.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        print(f"Saved script to {text_path}")
        results.append(("text", text_path))
        print("\nGENERATED SCRIPT:")
        print("-" * 40)
        print(text_content[:500] + ("..." if len(text_content) > 500 else ""))
        print("-" * 40)
    
    # Handle video content
    video_data = None
    if isinstance(response_data, dict) and "video_data" in response_data:
        video_data = response_data["video_data"]
    
    if video_data:
        video_path = os.path.join(data_dir, f"video_{request_id}.mp4")
        save_base64_to_file(video_data, video_path)
        results.append(("video", video_path))
    else:
        # Try media endpoint for video
        try:
            print("Trying to get video from media endpoint...")
            response = requests.get(f"{base_url}/api/v1/media/{request_id}")
            
            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "")
                if content_type.startswith("video/"):
                    video_path = os.path.join(data_dir, f"video_{request_id}.mp4")
                    with open(video_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Saved video from media endpoint to {video_path}")
                    results.append(("video", video_path))
        except Exception as e:
            print(f"Error getting video: {str(e)}")
    
    # Check for audio and image separately if video wasn't generated
    if not any(content_type == "video" for content_type, _ in results):
        # Check for audio
        audio_data = None
        if isinstance(response_data, dict) and "audio_data" in response_data:
            audio_data = response_data["audio_data"]
        
        if audio_data:
            audio_path = os.path.join(data_dir, f"audio_{request_id}.wav")
            save_base64_to_file(audio_data, audio_path)
            results.append(("audio", audio_path))
        
        # Check for image
        image_data = None
        if isinstance(response_data, dict) and "image_data" in response_data:
            image_data = response_data["image_data"]
        
        if image_data:
            image_path = os.path.join(data_dir, f"image_{request_id}.png")
            save_base64_to_file(image_data, image_path)
            results.append(("image", image_path))
    
    if results:
        return results
    else:
        print("No media content found in response")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test video reel generation")
    parser.add_argument("prompt", nargs="?", 
                      default="Create a 5-second fitness motivation reel that includes a short script about building strength, a voiceover, and images showing workout exercises.",
                      help="Prompt for video reel generation")
    parser.add_argument("--url", default="http://localhost:8000", 
                      help="Base URL for the API server")
    args = parser.parse_args()
    
    results = make_request(args.prompt, args.url)
    
    if results:
        print("\nSUCCESS: Video reel content generated and saved successfully!")
        # Try to open the generated files
        for content_type, file_path in results:
            try:
                if sys.platform == 'win32':
                    os.startfile(file_path)
                elif sys.platform == 'darwin':  # macOS
                    os.system(f'open "{file_path}"')
                else:  # Linux
                    os.system(f'xdg-open "{file_path}"')
            except Exception as e:
                print(f"Could not open {content_type} file automatically: {str(e)}")
    else:
        print("\nFAILURE: Could not generate or save video reel content.")
        sys.exit(1)