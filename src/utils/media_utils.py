"""
Utility functions for media processing.
"""

import os
import base64
from typing import List, Union, Optional, Tuple
import tempfile
from io import BytesIO
import subprocess
from PIL import Image
import requests


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string with data URI prefix
    """
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
        mime_type = get_mime_type(image_path)
        return f"data:{mime_type};base64,{encoded}"


def decode_base64_to_image(base64_str: str) -> Image.Image:
    """
    Decode a base64 string to a PIL Image.
    
    Args:
        base64_str: Base64 encoded string (with or without data URI prefix)
        
    Returns:
        PIL Image object
    """
    # Remove data URI prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))


def save_base64_to_file(base64_str: str, output_path: str) -> str:
    """
    Save a base64 encoded string to a file.
    
    Args:
        base64_str: Base64 encoded string (with or without data URI prefix)
        output_path: Path to save the file
        
    Returns:
        Path to the saved file
    """
    # Remove data URI prefix if present
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(base64_str))
    
    return output_path


def get_mime_type(file_path: str) -> str:
    """
    Get the MIME type of a file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MIME type string
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
    }
    
    return mime_types.get(extension, "application/octet-stream")


def download_image_from_url(url: str) -> Image.Image:
    """
    Download an image from a URL.
    
    Args:
        url: URL of the image
        
    Returns:
        PIL Image object
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        width: Target width
        height: Target height
        
    Returns:
        Resized PIL Image object
    """
    return image.resize((width, height), Image.LANCZOS)


def create_video_from_images(
    images: List[Union[str, Image.Image]], 
    output_path: str, 
    fps: int = 1,
    audio_path: Optional[str] = None
) -> str:
    """
    Create a video from a list of images.
    
    Args:
        images: List of image paths or PIL Image objects
        output_path: Path to save the output video
        fps: Frames per second
        audio_path: Optional path to an audio file to add to the video
        
    Returns:
        Path to the created video
    """
    # Create a temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save images as frames
        frame_paths = []
        for i, img in enumerate(images):
            if isinstance(img, str):
                # If it's a base64 string
                if img.startswith('data:image'):
                    image = decode_base64_to_image(img)
                # If it's a file path
                else:
                    image = Image.open(img)
            else:
                image = img
                
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            image.save(frame_path)
            frame_paths.append(frame_path)
        
        # Create video from frames
        temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-pattern_type', 'glob', '-i', os.path.join(temp_dir, 'frame_*.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 
            '-preset', 'fast', '-crf', '23',
            temp_video_path
        ]
        subprocess.run(cmd, check=True)
        
        # Add audio if provided
        if audio_path:
            cmd = [
                'ffmpeg', '-y', '-i', temp_video_path, '-i', audio_path,
                '-c:v', 'copy', '-c:a', 'aac', '-shortest',
                output_path
            ]
            subprocess.run(cmd, check=True)
        else:
            # Just copy the video if no audio
            with open(temp_video_path, 'rb') as src, open(output_path, 'wb') as dst:
                dst.write(src.read())
    
    return output_path


def extract_audio_from_video(video_path: str, output_path: str) -> str:
    """
    Extract audio from a video file.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted audio
        
    Returns:
        Path to the extracted audio
    """
    cmd = [
        'ffmpeg', '-y', '-i', video_path, 
        '-vn', '-acodec', 'copy', output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path


def extract_frames_from_video(
    video_path: str, 
    output_dir: str, 
    fps: Optional[float] = None,
    max_frames: Optional[int] = None
) -> List[str]:
    """
    Extract frames from a video.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the frames
        fps: Frames per second to extract (None for all frames)
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of paths to the extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the ffmpeg command
    cmd = ['ffmpeg', '-i', video_path]
    
    # Add fps filter if specified
    if fps is not None:
        cmd.extend(['-vf', f'fps={fps}'])
    
    # Add output pattern
    cmd.append(os.path.join(output_dir, 'frame_%04d.png'))
    
    # Run the command
    subprocess.run(cmd, check=True)
    
    # Get list of frames
    frames = sorted([
        os.path.join(output_dir, f) 
        for f in os.listdir(output_dir) 
        if f.startswith('frame_') and f.endswith('.png')
    ])
    
    # Limit number of frames if specified
    if max_frames is not None and len(frames) > max_frames:
        # Evenly distribute frame selection
        step = len(frames) / max_frames
        selected_frames = [frames[int(i * step)] for i in range(max_frames)]
        
        # Remove unselected frames
        for frame in frames:
            if frame not in selected_frames:
                os.remove(frame)
        
        frames = selected_frames
    
    return frames


def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video information (duration, width, height, fps)
    """
    # Run ffprobe to get video information
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration',
        '-of', 'json', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # Parse the JSON output
    import json
    info = json.loads(result.stdout)
    stream = info['streams'][0]
    
    # Calculate FPS from the fraction
    fps_parts = stream['r_frame_rate'].split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) > 1 else float(fps_parts[0])
    
    return {
        'width': stream.get('width', 0),
        'height': stream.get('height', 0),
        'fps': fps,
        'duration': float(stream.get('duration', 0))
    }


def combine_audio_tracks(
    audio_paths: List[str], 
    output_path: str,
    volumes: Optional[List[float]] = None
) -> str:
    """
    Combine multiple audio tracks into one.
    
    Args:
        audio_paths: List of audio file paths
        output_path: Path to save the combined audio
        volumes: Optional list of volume levels (0.0 to 1.0) for each track
        
    Returns:
        Path to the combined audio
    """
    if len(audio_paths) == 0:
        raise ValueError("No audio paths provided")
    
    if len(audio_paths) == 1:
        # Just copy the single audio file
        with open(audio_paths[0], 'rb') as src, open(output_path, 'wb') as dst:
            dst.write(src.read())
        return output_path
    
    # Prepare the filter complex for mixing
    filter_parts = []
    inputs = []
    
    for i, path in enumerate(audio_paths):
        inputs.extend(['-i', path])
        
        # Add volume filter if specified
        if volumes and i < len(volumes):
            filter_parts.append(f'[{i}]volume={volumes[i]}[a{i}]')
        else:
            filter_parts.append(f'[{i}]')
    
    # Create the amix filter
    mix_parts = ''.join(f'[a{i}]' if volumes else f'[{i}]' for i in range(len(audio_paths)))
    filter_parts.append(f'{mix_parts}amix=inputs={len(audio_paths)}:duration=longest[out]')
    
    # Build the full command
    cmd = [
        'ffmpeg', '-y'
    ] + inputs + [
        '-filter_complex', ';'.join(filter_parts),
        '-map', '[out]',
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    return output_path


def overlay_text_on_image(
    image: Union[str, Image.Image], 
    text: str,
    output_path: str,
    position: Tuple[int, int] = None,
    font_size: int = 30,
    color: str = "white",
    stroke_color: str = "black",
    stroke_width: int = 2
) -> str:
    """
    Overlay text on an image.
    
    Args:
        image: Image file path or PIL Image object
        text: Text to overlay
        output_path: Path to save the output image
        position: (x, y) position for the text (None for centered)
        font_size: Font size
        color: Text color
        stroke_color: Outline color
        stroke_width: Outline width
        
    Returns:
        Path to the output image
    """
    # Load the image if it's a path
    if isinstance(image, str):
        if image.startswith('data:image'):
            img = decode_base64_to_image(image)
        else:
            img = Image.open(image)
    else:
        img = image
    
    # Use ImageDraw to add text
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text position if not provided
    if position is None:
        text_width, text_height = draw.textsize(text, font=font)
        position = ((img.width - text_width) // 2, (img.height - text_height) // 2)
    
    # Draw the text with stroke
    draw.text(position, text, font=font, fill=color, stroke_width=stroke_width, stroke_fill=stroke_color)
    
    # Save the result
    img.save(output_path)
    return output_path