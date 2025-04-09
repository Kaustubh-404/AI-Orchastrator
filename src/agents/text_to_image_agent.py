# src/agents/text_to_image_agent.py
import time
import uuid
import os
from typing import Dict, List, Optional, Any
import base64
from io import BytesIO
import torch
from PIL import Image, ImageDraw

from .base_agent import BaseAgent

# Try to import diffusers, but have a fallback
try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("Warning: diffusers package not available, using fallback image generation")
    DIFFUSERS_AVAILABLE = False

class TextToImageAgent(BaseAgent):
    """Agent for text-to-image generation tasks."""
    
    def __init__(self, 
                name: str = "Text-to-Image Agent", 
                description: str = "Generates images from text descriptions",
                agent_id: Optional[str] = None,
                model_name: str = "runwayml/stable-diffusion-v1-5"):
        """
        Initialize the text-to-image agent.
        """
        if agent_id is None:
            agent_id = str(uuid.uuid4())
        super().__init__(agent_id, name, description)
        
        # Initialize the model name but don't load the model yet
        self.model_name = model_name
        print(f"Initializing TextToImageAgent with model: {model_name}")
        
        # Try to use CUDA if available
        self.use_cuda = False
        if torch.cuda.is_available():
            self.device = "cuda"
            self.use_cuda = True
            print(f"CUDA is available. Using device: {self.device}")
        else:
            self.device = "cpu"
            print(f"CUDA is not available. Using device: {self.device}")
        
        # Check if we should use the lightweight model setting
        use_lightweight = os.environ.get("LIGHTWEIGHT_MODEL", "false").lower() == "true"
        if use_lightweight:
            print("Using lightweight model configuration")
            self.model_name = "CompVis/stable-diffusion-v1-4"
        
        # Initialize but don't load the model yet
        self.pipe = None
        self.model_loaded = False
    
    def load_model(self):
        # Load the model only when needed
        if DIFFUSERS_AVAILABLE and not self.model_loaded:
            try:
                print(f"Loading model: {self.model_name}")
                
                # Load with low memory usage
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    self.model_name,
                    safety_checker=None  # Disable for speed
                )
                
                # Apply memory optimizations
                if self.use_cuda:
                    self.pipe = self.pipe.to(self.device)
                    self.pipe.enable_attention_slicing()
                else:
                    # CPU optimizations
                    self.pipe.enable_attention_slicing()
                
                print("Model loaded successfully")
                self.model_loaded = True
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Will use fallback image generation")
                self.model_loaded = False

    def unload_model(self):
        # Keep the model loaded for image generation
        # Text-to-image is our main GPU workload, so we want to keep it loaded
        # but free up other CUDA resources
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    
    def get_capabilities(self) -> List[str]:
        """Return a list of capabilities this agent provides."""
        return [
            "text_to_image",
            "image_generation"
        ]
    
    async def process(self, task_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a text-to-image task.
        """
        start_time = time.time()
        success = False
        result = {}
        
        try:
            # Load the model if needed
            self.load_model()
            
            # Extract parameters
            prompt = task_data.get("prompt", task_data.get("description", ""))
            if not prompt:
                raise ValueError("No prompt provided")
            
            print(f"Processing image generation for prompt: {prompt}")
            
            # Use smaller dimensions when not using CUDA
            width = task_data.get("width", 384 if not self.use_cuda else 512)
            height = task_data.get("height", 384 if not self.use_cuda else 512)
            
            # Use fewer steps when not using CUDA
            num_inference_steps = task_data.get("num_inference_steps", 
                                               20 if not self.use_cuda else 30)
            
            guidance_scale = task_data.get("guidance_scale", 7.5)
            negative_prompt = task_data.get("negative_prompt", None)
            
            # Generate the image
            if self.model_loaded and DIFFUSERS_AVAILABLE:
                try:
                    print(f"Generating image with {self.model_name}")
                    print(f"Parameters: {width}x{height}, {num_inference_steps} steps")
                    
                    # Generate the image
                    image = self.pipe(
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        negative_prompt=negative_prompt
                    ).images[0]
                    
                    print("Image generated successfully with model")
                except Exception as e:
                    print(f"Error generating image with model: {str(e)}")
                    print("Falling back to simple image generation")
                    image = self._generate_simple_image(prompt, width, height)
            else:
                print("Using simple image generation")
                image = self._generate_simple_image(prompt, width, height)
            
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
            error_msg = str(e)
            print(f"Text-to-image processing error: {error_msg}")
            result = {
                "error": error_msg,
                "task_id": task_data.get("task_id"),
                "status": "failed",
            }
        
        # Don't unload the model but clean up cache
        self.unload_model()
        
        # Update metrics
        processing_time = time.time() - start_time
        self.update_metrics(success, processing_time)
        
        return result
    
    def _generate_simple_image(self, prompt: str, width: int = 512, height: int = 512) -> Image.Image:
        """Generate a simple image based on the prompt."""
        import hashlib
        import random
        
        print(f"Generating simple image for prompt: {prompt}")
        
        # Use the prompt to deterministically generate colors
        hash_obj = hashlib.md5(prompt.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        random.seed(hash_int)
        
        # Create a base color based on the prompt
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        
        # Create a gradient background
        img = Image.new('RGB', (width, height), (r, g, b))
        draw = ImageDraw.Draw(img)
        
        # Add some shapes based on words in the prompt
        words = prompt.split()
        for word in words:
            try:
                shape_type = hash(word) % 3  # 0: rectangle, 1: circle, 2: line
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                x2 = random.randint(0, width)
                y2 = random.randint(0, height)
                fill_r = (r + random.randint(-50, 50)) % 256
                fill_g = (g + random.randint(-50, 50)) % 256
                fill_b = (b + random.randint(-50, 50)) % 256
                fill = (fill_r, fill_g, fill_b)
                
                if shape_type == 0:
                    draw.rectangle([x1, y1, x2, y2], fill=fill)
                elif shape_type == 1:
                    radius = abs(x2 - x1) // 2
                    draw.ellipse([x1, y1, x1 + radius, y1 + radius], fill=fill)
                else:
                    draw.line([x1, y1, x2, y2], fill=fill, width=5)
            except Exception as e:
                print(f"Error drawing shape for word '{word}': {e}")
        
        # Add a text caption
        try:
            draw.text((10, height - 20), f"AI: {prompt[:20]}...", fill="white")
        except Exception as e:
            print(f"Error adding caption: {e}")
        
        return img




# # src/agents/text_to_image_agent.py
# import time
# import uuid
# import os
# from typing import Dict, List, Optional, Any
# import base64
# from io import BytesIO
# import torch
# from PIL import Image, ImageDraw

# from .base_agent import BaseAgent

# # Try to import diffusers, but have a fallback
# try:
#     from diffusers import StableDiffusionPipeline
#     DIFFUSERS_AVAILABLE = True
# except ImportError:
#     print("Warning: diffusers package not available, using fallback image generation")
#     DIFFUSERS_AVAILABLE = False

# class TextToImageAgent(BaseAgent):
#     """Agent for text-to-image generation tasks."""
    
#     def __init__(self, 
#                 name: str = "Text-to-Image Agent", 
#                 description: str = "Generates images from text descriptions",
#                 agent_id: Optional[str] = None,
#                 model_name: str = "runwayml/stable-diffusion-v1-5"):
#         """
#         Initialize the text-to-image agent.
#         """
#         if agent_id is None:
#             agent_id = str(uuid.uuid4())
#         super().__init__(agent_id, name, description)
        
#         # Initialize the model
#         self.model_name = model_name
#         print(f"Initializing TextToImageAgent with model: {model_name}")
        
#         # Try to use CUDA if available
#         self.use_cuda = False
#         if torch.cuda.is_available():
#             self.device = "cuda"
#             self.use_cuda = True
#             print(f"CUDA is available. Using device: {self.device}")
#         else:
#             self.device = "cpu"
#             self.model_loaded = False
#             print(f"CUDA is not available. Using device: {self.device}")
        
#         # Check if we should use the lightweight model setting
#         use_lightweight = os.environ.get("LIGHTWEIGHT_MODEL", "false").lower() == "true"
#         if use_lightweight:
#             print("Using lightweight model configuration")
            
#         # Initialize model
#         self.model_loaded = False
#         if DIFFUSERS_AVAILABLE:
#             try:
#                 # Use the smallest/fastest model when lightweight is enabled
#                 if use_lightweight:
#                     self.model_name = "CompVis/stable-diffusion-v1-4"
                
#                 print(f"Loading model: {self.model_name}")
                
#                 # Load with low memory usage
#                 self.pipe = StableDiffusionPipeline.from_pretrained(
#                     self.model_name,
#                     safety_checker=None  # Disable for speed
#                 )
                
#                 # Apply memory optimizations
#                 if self.use_cuda:
#                     self.pipe = self.pipe.to(self.device)
#                     self.pipe.enable_attention_slicing()
#                 else:
#                     # CPU optimizations
#                     self.pipe.enable_attention_slicing()
                
#                 print("Model loaded successfully")
#                 self.model_loaded = True
#             except Exception as e:
#                 print(f"Error loading model: {str(e)}")
#                 print("Will use fallback image generation")
    
#     def get_capabilities(self) -> List[str]:
#         """Return a list of capabilities this agent provides."""
#         return [
#             "text_to_image",
#             "image_generation"
#         ]
    
#     async def process(self, task_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """
#         Process a text-to-image task.
#         """
#         start_time = time.time()
#         success = False
#         result = {}
        
#         try:
#             # Extract parameters
#             prompt = task_data.get("prompt", task_data.get("description", ""))
#             if not prompt:
#                 raise ValueError("No prompt provided")
            
#             print(f"Processing image generation for prompt: {prompt}")
            
#             # Use smaller dimensions when not using CUDA
#             width = task_data.get("width", 384 if not self.use_cuda else 512)
#             height = task_data.get("height", 384 if not self.use_cuda else 512)
            
#             # Use fewer steps when not using CUDA
#             num_inference_steps = task_data.get("num_inference_steps", 
#                                                20 if not self.use_cuda else 30)
            
#             guidance_scale = task_data.get("guidance_scale", 7.5)
#             negative_prompt = task_data.get("negative_prompt", None)
            
#             # Generate the image
#             if self.model_loaded and DIFFUSERS_AVAILABLE:
#                 try:
#                     print(f"Generating image with {self.model_name}")
#                     print(f"Parameters: {width}x{height}, {num_inference_steps} steps")
                    
#                     # Generate the image
#                     image = self.pipe(
#                         prompt=prompt,
#                         height=height,
#                         width=width,
#                         num_inference_steps=num_inference_steps,
#                         guidance_scale=guidance_scale,
#                         negative_prompt=negative_prompt
#                     ).images[0]
                    
#                     print("Image generated successfully with model")
#                 except Exception as e:
#                     print(f"Error generating image with model: {str(e)}")
#                     print("Falling back to simple image generation")
#                     image = self._generate_simple_image(prompt, width, height)
#             else:
#                 print("Using simple image generation")
#                 image = self._generate_simple_image(prompt, width, height)
            
#             # Convert to base64
#             buffered = BytesIO()
#             image.save(buffered, format="PNG")
#             img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
#             result = {
#                 "image_data": f"data:image/png;base64,{img_str}",
#                 "task_id": task_data.get("task_id"),
#                 "status": "completed",
#             }
#             success = True
            
#         except Exception as e:
#             error_msg = str(e)
#             print(f"Text-to-image processing error: {error_msg}")
#             result = {
#                 "error": error_msg,
#                 "task_id": task_data.get("task_id"),
#                 "status": "failed",
#             }
        
#         # Update metrics
#         processing_time = time.time() - start_time
#         self.update_metrics(success, processing_time)
        
#         return result
    
#     def _generate_simple_image(self, prompt: str, width: int = 512, height: int = 512) -> Image.Image:
#         """Generate a simple image based on the prompt."""
#         import hashlib
#         import random
        
#         print(f"Generating simple image for prompt: {prompt}")
        
#         # Use the prompt to deterministically generate colors
#         hash_obj = hashlib.md5(prompt.encode())
#         hash_int = int(hash_obj.hexdigest(), 16)
#         random.seed(hash_int)
        
#         # Create a base color based on the prompt
#         r = random.randint(0, 255)
#         g = random.randint(0, 255)
#         b = random.randint(0, 255)
        
#         # Create a gradient background
#         img = Image.new('RGB', (width, height), (r, g, b))
#         draw = ImageDraw.Draw(img)
        
#         # Add some shapes based on words in the prompt
#         words = prompt.split()
#         for word in words:
#             try:
#                 shape_type = hash(word) % 3  # 0: rectangle, 1: circle, 2: line
#                 x1 = random.randint(0, width)
#                 y1 = random.randint(0, height)
#                 x2 = random.randint(0, width)
#                 y2 = random.randint(0, height)
#                 fill_r = (r + random.randint(-50, 50)) % 256
#                 fill_g = (g + random.randint(-50, 50)) % 256
#                 fill_b = (b + random.randint(-50, 50)) % 256
#                 fill = (fill_r, fill_g, fill_b)
                
#                 if shape_type == 0:
#                     draw.rectangle([x1, y1, x2, y2], fill=fill)
#                 elif shape_type == 1:
#                     radius = abs(x2 - x1) // 2
#                     draw.ellipse([x1, y1, x1 + radius, y1 + radius], fill=fill)
#                 else:
#                     draw.line([x1, y1, x2, y2], fill=fill, width=5)
#             except Exception as e:
#                 print(f"Error drawing shape for word '{word}': {e}")
        
#         # Add a text caption
#         try:
#             draw.text((10, height - 20), f"AI: {prompt[:20]}...", fill="white")
#         except Exception as e:
#             print(f"Error adding caption: {e}")
        
#         return img


# # # src/agents/text_to_image_agent.py
# # import time
# # import uuid
# # import os
# # from typing import Dict, List, Optional, Any
# # import base64
# # from io import BytesIO
# # from diffusers import StableDiffusionPipeline
# # import torch

# # from .base_agent import BaseAgent

# # # class TextToImageAgent(BaseAgent):
# # #     """Agent for text-to-image generation tasks."""
    
# # #     def __init__(self, 
# # #                 name: str = "Text-to-Image Agent", 
# # #                 description: str = "Generates images from text descriptions",
# # #                 agent_id: Optional[str] = None,
# # #                 model_name: str = "runwayml/stable-diffusion-v1-5"):
# # #         """
# # #         Initialize the text-to-image agent.
        
# # #         Args:
# # #             name: Name of the agent
# # #             description: Description of the agent
# # #             agent_id: Unique ID for the agent (generated if not provided)
# # #             model_name: Hugging Face model to use
# # #         """
# # #         if agent_id is None:
# # #             agent_id = str(uuid.uuid4())
# # #         super().__init__(agent_id, name, description)
        
# # #         # Initialize the model
# # #         self.model_name = model_name
        
# # #         # Use CUDA if available
# # #         self.device = "cuda" if torch.cuda.is_available() else "cpu"
# # #         self.pipe = StableDiffusionPipeline.from_pretrained(model_name)
# # #         if self.device == "cuda":
# # #             self.pipe = self.pipe.to(self.device)

# # class TextToImageAgent(BaseAgent):
# #     """Agent for text-to-image generation tasks."""
    
# #     def __init__(self, 
# #                 name: str = "Text-to-Image Agent", 
# #                 description: str = "Generates images from text descriptions",
# #                 agent_id: Optional[str] = None,
# #                 model_name: str = "runwayml/stable-diffusion-v1-5"):
# #         """
# #         Initialize the text-to-image agent.
# #         """
# #         if agent_id is None:
# #             agent_id = str(uuid.uuid4())
# #         super().__init__(agent_id, name, description)
        
# #         # Initialize the model
# #         self.model_name = model_name
# #         print(f"Initializing TextToImageAgent with model {model_name}")
        
# #         # Use CUDA if available
# #         self.device = "cuda" if torch.cuda.is_available() else "cpu"
# #         print(f"Using device: {self.device}")
        
# #         try:
# #             # Try to load the model with memory optimization
# #             self.pipe = StableDiffusionPipeline.from_pretrained(
# #                 model_name,
# #                 torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
# #                 safety_checker=None  # Disable safety checker for faster loading
# #             )
# #             if self.device == "cuda":
# #                 self.pipe = self.pipe.to(self.device)
# #             print("Model loaded successfully")
# #         except Exception as e:
# #             print(f"Error loading model: {str(e)}")
# #             # Try a smaller model as fallback
# #             try:
# #                 self.model_name = "CompVis/stable-diffusion-v1-4"
# #                 print(f"Trying smaller model: {self.model_name}")
# #                 self.pipe = StableDiffusionPipeline.from_pretrained(
# #                     self.model_name,
# #                     torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
# #                     safety_checker=None
# #                 )
# #                 if self.device == "cuda":
# #                     self.pipe = self.pipe.to(self.device)
# #                 print("Fallback model loaded successfully")
# #             except Exception as e2:
# #                 print(f"Error loading fallback model: {str(e2)}")
# #                 # Set a flag to indicate model loading failed
# #                 self.model_loaded = False
# #                 return
        
# #         self.model_loaded = True
    
# #     def get_capabilities(self) -> List[str]:
# #         """Return a list of capabilities this agent provides."""
# #         return [
# #             "text_to_image",
# #             "image_generation"
# #         ]

    
# #     async def process(self, task_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
# #         """Process a text-to-image task."""
# #         start_time = time.time()
# #         success = False
# #         result = {}
        
# #         try:
# #             # Check if model is loaded
# #             if not getattr(self, 'model_loaded', False):
# #                 raise ValueError("Model failed to load. Cannot generate images.")
                
# #             # Extract parameters
# #             prompt = task_data.get("prompt")
# #             if not prompt:
# #                 raise ValueError("No prompt provided")
            
# #             print(f"Generating image for prompt: {prompt}")
            
# #             # Optional parameters with smaller defaults to reduce memory usage
# #             width = task_data.get("width", 512)
# #             height = task_data.get("height", 512)
# #             num_inference_steps = task_data.get("num_inference_steps", 30)  # Reduced from 50
# #             guidance_scale = task_data.get("guidance_scale", 7.5)
# #             negative_prompt = task_data.get("negative_prompt", None)
            
# #             # Generate the image
# #             try:
# #                 image = self.pipe(
# #                     prompt=prompt,
# #                     height=height,
# #                     width=width,
# #                     num_inference_steps=num_inference_steps,
# #                     guidance_scale=guidance_scale,
# #                     negative_prompt=negative_prompt
# #                 ).images[0]
# #                 print("Image generated successfully")
# #             except Exception as e:
# #                 print(f"Error during image generation: {str(e)}")
# #                 # Try again with even smaller parameters
# #                 try:
# #                     print("Retrying with smaller parameters")
# #                     image = self.pipe(
# #                         prompt=prompt,
# #                         height=256,
# #                         width=256,
# #                         num_inference_steps=20,
# #                         guidance_scale=7.0,
# #                     ).images[0]
# #                     print("Image generated successfully on retry")
# #                 except Exception as e2:
# #                     print(f"Error during retry: {str(e2)}")
# #                     raise
            
# #             # Convert to base64
# #             buffered = BytesIO()
# #             image.save(buffered, format="PNG")
# #             img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
# #             result = {
# #                 "image_data": f"data:image/png;base64,{img_str}",
# #                 "task_id": task_data.get("task_id"),
# #                 "status": "completed",
# #             }
# #             success = True
            
# #         except Exception as e:
# #             error_msg = str(e)
# #             print(f"Text-to-image processing error: {error_msg}")
# #             result = {
# #                 "error": error_msg,
# #                 "task_id": task_data.get("task_id"),
# #                 "status": "failed",
# #             }
        
# #         # Update metrics
# #         processing_time = time.time() - start_time
# #         self.update_metrics(success, processing_time)
        
# #         return result
    
# #     # async def process(self, task_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
# #     #     """
# #     #     Process a text-to-image task.
        
# #     #     Args:
# #     #         task_data: Contains 'prompt' and generation parameters
# #     #         context: Optional additional context
            
# #     #     Returns:
# #     #         Dict containing the generated image as base64
# #     #     """
# #     #     start_time = time.time()
# #     #     success = False
# #     #     result = {}
        
# #     #     try:
# #     #         # Extract parameters
# #     #         prompt = task_data.get("prompt")
# #     #         if not prompt:
# #     #             raise ValueError("No prompt provided")
            
# #     #         # Optional parameters
# #     #         width = task_data.get("width", 512)
# #     #         height = task_data.get("height", 512)
# #     #         num_inference_steps = task_data.get("num_inference_steps", 50)
# #     #         guidance_scale = task_data.get("guidance_scale", 7.5)
# #     #         negative_prompt = task_data.get("negative_prompt", None)
            
# #     #         # Generate the image
# #     #         image = self.pipe(
# #     #             prompt=prompt,
# #     #             height=height,
# #     #             width=width,
# #     #             num_inference_steps=num_inference_steps,
# #     #             guidance_scale=guidance_scale,
# #     #             negative_prompt=negative_prompt
# #     #         ).images[0]
            
# #     #         # Convert to base64
# #     #         buffered = BytesIO()
# #     #         image.save(buffered, format="PNG")
# #     #         img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
# #     #         result = {
# #     #             "image_data": f"data:image/png;base64,{img_str}",
# #     #             "task_id": task_data.get("task_id"),
# #     #             "status": "completed",
# #     #         }
# #     #         success = True
            
# #     #     except Exception as e:
# #     #         result = {
# #     #             "error": str(e),
# #     #             "task_id": task_data.get("task_id"),
# #     #             "status": "failed",
# #     #         }
        
# #     #     # Update metrics
# #     #     processing_time = time.time() - start_time
# #     #     self.update_metrics(success, processing_time)
        
# #     #     return result