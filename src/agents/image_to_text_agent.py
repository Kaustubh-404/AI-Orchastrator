# src/agents/image_to_text_agent.py
import time
import uuid
import os
from typing import Dict, List, Optional, Any
import base64
from io import BytesIO
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from .base_agent import BaseAgent

class ImageToTextAgent(BaseAgent):
    """Agent for image-to-text conversion tasks."""
    
    def __init__(self, 
                name: str = "Image-to-Text Agent", 
                description: str = "Converts images to textual descriptions",
                agent_id: Optional[str] = None,
                model_name: str = "Salesforce/blip-image-captioning-large"):
        """
        Initialize the image-to-text agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent
            agent_id: Unique ID for the agent (generated if not provided)
            model_name: Hugging Face model to use
        """
        if agent_id is None:
            agent_id = str(uuid.uuid4())
        super().__init__(agent_id, name, description)
        
        # Initialize just the model name but don't load the model yet
        self.model_name = model_name
        self.device = "cpu"  # Force CPU to save GPU memory
        self.processor = None
        self.model = None
    
    def load_model(self):
        # Load the model only when needed
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name).to(self.device)

    def unload_model(self):
        # Release the model when done
        self.model = None
        self.processor = None
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_capabilities(self) -> List[str]:
        """Return a list of capabilities this agent provides."""
        return [
            "image_to_text",
            "image_captioning",
            "image_description"
        ]
    
    async def process(self, task_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an image-to-text task.
        
        Args:
            task_data: Contains 'image_data' (base64 or URL) and options
            context: Optional additional context
            
        Returns:
            Dict containing the text description of the image
        """
        start_time = time.time()
        success = False
        result = {}
        
        try:
            # Load the model at the start
            self.load_model()
            
            # Extract parameters
            image_data = task_data.get("image_data")
            if not image_data:
                raise ValueError("No image data provided")
            
            # Load the image
            image = self._load_image(image_data)
            
            # Process the image
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            text_description = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            result = {
                "text_description": text_description,
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
        
        finally:
            # Unload the model when finished
            self.unload_model()
        
        # Update metrics
        processing_time = time.time() - start_time
        self.update_metrics(success, processing_time)
        
        return result
    
    def _load_image(self, image_data):
        """
        Load an image from either a base64 string or URL.
        
        Args:
            image_data: Base64 string or URL to an image
            
        Returns:
            PIL Image object
        """
        # Check if it's a URL
        if isinstance(image_data, str) and (image_data.startswith('http://') or image_data.startswith('https://')):
            response = requests.get(image_data)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        
        # Check if it's a base64 string
        elif isinstance(image_data, str) and image_data.startswith('data:image'):
            # Extract the base64 part
            base64_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
            return Image.open(BytesIO(image_bytes))
        
        # Assume it's a file path
        elif isinstance(image_data, str):
            return Image.open(image_data)
        
        # Assume it's already a PIL Image
        elif isinstance(image_data, Image.Image):
            return image_data
        
        else:
            raise ValueError("Unsupported image data format")




# # src/agents/image_to_text_agent.py
# import time
# import uuid
# import os
# from typing import Dict, List, Optional, Any
# import base64
# from io import BytesIO
# import requests
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForVision2Seq

# from .base_agent import BaseAgent

# class ImageToTextAgent(BaseAgent):
#     """Agent for image-to-text conversion tasks."""
    
#     def __init__(self, 
#                 name: str = "Image-to-Text Agent", 
#                 description: str = "Converts images to textual descriptions",
#                 agent_id: Optional[str] = None,
#                 model_name: str = "Salesforce/blip-image-captioning-large"):
#         """
#         Initialize the image-to-text agent.
        
#         Args:
#             name: Name of the agent
#             description: Description of the agent
#             agent_id: Unique ID for the agent (generated if not provided)
#             model_name: Hugging Face model to use
#         """
#         if agent_id is None:
#             agent_id = str(uuid.uuid4())
#         super().__init__(agent_id, name, description)
        
#         # Initialize the model
#         self.model_name = model_name
#         self.processor = AutoProcessor.from_pretrained(model_name)
#         self.model = AutoModelForVision2Seq.from_pretrained(model_name)
    
#     def get_capabilities(self) -> List[str]:
#         """Return a list of capabilities this agent provides."""
#         return [
#             "image_to_text",
#             "image_captioning",
#             "image_description"
#         ]
    
#     async def process(self, task_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """
#         Process an image-to-text task.
        
#         Args:
#             task_data: Contains 'image_data' (base64 or URL) and options
#             context: Optional additional context
            
#         Returns:
#             Dict containing the text description of the image
#         """
#         start_time = time.time()
#         success = False
#         result = {}
        
#         try:
#             # Extract parameters
#             image_data = task_data.get("image_data")
#             if not image_data:
#                 raise ValueError("No image data provided")
            
#             # Load the image
#             image = self._load_image(image_data)
            
#             # Process the image
#             inputs = self.processor(images=image, return_tensors="pt")
#             outputs = self.model.generate(**inputs)
#             text_description = self.processor.decode(outputs[0], skip_special_tokens=True)
            
#             result = {
#                 "text_description": text_description,
#                 "task_id": task_data.get("task_id"),
#                 "status": "completed",
#             }
#             success = True
            
#         except Exception as e:
#             result = {
#                 "error": str(e),
#                 "task_id": task_data.get("task_id"),
#                 "status": "failed",
#             }
        
#         # Update metrics
#         processing_time = time.time() - start_time
#         self.update_metrics(success, processing_time)
        
#         return result
    
#     def _load_image(self, image_data):
#         """
#         Load an image from either a base64 string or URL.
        
#         Args:
#             image_data: Base64 string or URL to an image
            
#         Returns:
#             PIL Image object
#         """
#         # Check if it's a URL
#         if isinstance(image_data, str) and (image_data.startswith('http://') or image_data.startswith('https://')):
#             response = requests.get(image_data)
#             response.raise_for_status()
#             return Image.open(BytesIO(response.content))
        
#         # Check if it's a base64 string
#         elif isinstance(image_data, str) and image_data.startswith('data:image'):
#             # Extract the base64 part
#             base64_data = image_data.split(',')[1]
#             image_bytes = base64.b64decode(base64_data)
#             return Image.open(BytesIO(image_bytes))
        
#         # Assume it's a file path
#         elif isinstance(image_data, str):
#             return Image.open(image_data)
        
#         # Assume it's already a PIL Image
#         elif isinstance(image_data, Image.Image):
#             return image_data
        
#         else:
#             raise ValueError("Unsupported image data format")