# src/agents/video_creation_agent.py
import time
import uuid
import os
from typing import Dict, List, Optional, Any
import base64
import tempfile
import subprocess
from PIL import Image
from io import BytesIO

from .base_agent import BaseAgent

class VideoCreationAgent(BaseAgent):
    """Agent for creating videos from images and audio."""
    
    def __init__(self, 
                name: str = "Video Creation Agent", 
                description: str = "Creates videos from images and audio",
                agent_id: Optional[str] = None):
        """
        Initialize the video creation agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent
            agent_id: Unique ID for the agent (generated if not provided)
        """
        if agent_id is None:
            agent_id = str(uuid.uuid4())
        super().__init__(agent_id, name, description)
        
        # Verify that ffmpeg is installed
        try:
            subprocess.check_output(['ffmpeg', '-version'])
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Warning: ffmpeg is not installed or not in PATH. Video creation may fail.")
    
    def get_capabilities(self) -> List[str]:
        """Return a list of capabilities this agent provides."""
        return [
            "video_creation",
            "image_to_video",
            "slideshow_creation"
        ]
    
    async def process(self, task_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a video creation task.
        
        Args:
            task_data: Contains 'images' (list of base64 strings) and 'audio' (optional base64 string)
            context: Optional additional context
            
        Returns:
            Dict containing the created video as base64
        """
        start_time = time.time()
        success = False
        result = {}
        
        try:
            # Extract parameters
            images = task_data.get("images", [])
            audio = task_data.get("audio")
            fps = task_data.get("fps", 1)
            
            if not images:
                raise ValueError("No images provided")
            
            # Create temporary directory for files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save images
                image_paths = []
                for i, img_data in enumerate(images):
                    img = self._load_image(img_data)
                    img_path = os.path.join(temp_dir, f"image_{i:04d}.png")
                    img.save(img_path)
                    image_paths.append(img_path)
                
                # Create video from images
                video_path = os.path.join(temp_dir, "output.mp4")
                self._create_video_from_images(image_paths, video_path, fps)
                
                # Add audio if provided
                final_video_path = video_path
                if audio:
                    audio_path = os.path.join(temp_dir, "audio.wav")
                    with open(audio_path, 'wb') as f:
                        if isinstance(audio, str) and audio.startswith('data:audio'):
                            audio_bytes = base64.b64decode(audio.split(',')[1])
                            f.write(audio_bytes)
                        else:
                            f.write(audio)
                    
                    # Combine video and audio
                    final_video_path = os.path.join(temp_dir, "output_with_audio.mp4")
                    self._add_audio_to_video(video_path, audio_path, final_video_path)
                
                # Read the final video
                with open(final_video_path, 'rb') as f:
                    video_bytes = f.read()
                
                # Convert to base64
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                
                result = {
                    "video_data": f"data:video/mp4;base64,{video_base64}",
                    "task_id": task_data.get("task_id"),
                    "status": "completed",
                }
                success = True
            
        except Exception as e:
            result = {
                "error": str(e),
                "task_id": task_data.get("task_id"),
                "status": "failed",
            }
        
        # Update metrics
        processing_time = time.time() - start_time
        self.update_metrics(success, processing_time)
        
        return result
    
    def _load_image(self, image_data):
        """Load image from various formats."""
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Extract the base64 part
            base64_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
            return Image.open(BytesIO(image_bytes))
        elif isinstance(image_data, str) and os.path.exists(image_data):
            return Image.open(image_data)
        elif isinstance(image_data, Image.Image):
            return image_data
        else:
            raise ValueError("Unsupported image data format")
    
    def _create_video_from_images(self, image_paths, output_path, fps):
        """Create a video from a list of images using ffmpeg."""
        cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-pattern_type', 'glob', '-i', os.path.join(os.path.dirname(image_paths[0]), 'image_*.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_path
        ]
        subprocess.check_call(cmd)
    
    def _add_audio_to_video(self, video_path, audio_path, output_path):
        """Add audio to a video using ffmpeg."""
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'aac', '-shortest', output_path
        ]
        subprocess.check_call(cmd)