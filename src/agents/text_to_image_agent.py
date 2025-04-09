# src/agents/text_to_image_agent.py
import time
import uuid
import os
from typing import Dict, List, Optional, Any
import base64
from io import BytesIO
from diffusers import StableDiffusionPipeline
import torch

from .base_agent import BaseAgent

class TextToImageAgent(BaseAgent):
    """Agent for text-to-image generation tasks."""
    
    def __init__(self, 
                name: str = "Text-to-Image Agent", 
                description: str = "Generates images from text descriptions",
                agent_id: Optional[str] = None,
                model_name: str = "runwayml/stable-diffusion-v1-5"):
        """
        Initialize the text-to-image agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent
            agent_id: Unique ID for the agent (generated if not provided)
            model_name: Hugging Face model to use
        """
        if agent_id is None:
            agent_id = str(uuid.uuid4())
        super().__init__(agent_id, name, description)
        
        # Initialize the model
        self.model_name = model_name
        
        # Use CUDA if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name)
        if self.device == "cuda":
            self.pipe = self.pipe.to(self.device)
    
    def get_capabilities(self) -> List[str]:
        """Return a list of capabilities this agent provides."""
        return [
            "text_to_image",
            "image_generation"
        ]
    
    async def process(self, task_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a text-to-image task.
        
        Args:
            task_data: Contains 'prompt' and generation parameters
            context: Optional additional context
            
        Returns:
            Dict containing the generated image as base64
        """
        start_time = time.time()
        success = False
        result = {}
        
        try:
            # Extract parameters
            prompt = task_data.get("prompt")
            if not prompt:
                raise ValueError("No prompt provided")
            
            # Optional parameters
            width = task_data.get("width", 512)
            height = task_data.get("height", 512)
            num_inference_steps = task_data.get("num_inference_steps", 50)
            guidance_scale = task_data.get("guidance_scale", 7.5)
            negative_prompt = task_data.get("negative_prompt", None)
            
            # Generate the image
            image = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt
            ).images[0]
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            result = {
                "image_data": f"data:image/png;base64,{img_str}",
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