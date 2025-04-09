# src/agents/text_to_audio_agent.py
import time
import uuid
import os
from typing import Dict, List, Optional, Any
import base64
from io import BytesIO
import torch
from transformers import AutoProcessor, AutoModel

from .base_agent import BaseAgent

class TextToAudioAgent(BaseAgent):
    """Agent for text-to-audio generation tasks."""
    
    def __init__(self, 
                name: str = "Text-to-Audio Agent", 
                description: str = "Generates speech from text",
                agent_id: Optional[str] = None,
                model_name: str = "facebook/mms-tts-eng"):
        """
        Initialize the text-to-audio agent.
        
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
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

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
            "text_to_audio",
            "text_to_speech",
            "voice_generation"
        ]
    
    async def process(self, task_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a text-to-audio task.
        
        Args:
            task_data: Contains 'text' and voice parameters
            context: Optional additional context
            
        Returns:
            Dict containing the generated audio as base64
        """
        start_time = time.time()
        success = False
        result = {}
        
        try:
            # Load the model at the start
            self.load_model()
            
            # Extract parameters
            text = task_data.get("text")
            if not text:
                raise ValueError("No text provided")
            
            # Generate the audio
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs)
            
            # Convert to base64
            audio_data = base64.b64encode(output.cpu().numpy().tobytes()).decode("utf-8")
            
            result = {
                "audio_data": f"data:audio/wav;base64,{audio_data}",
                "text": text,
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




# # src/agents/text_to_audio_agent.py
# import time
# import uuid
# import os
# from typing import Dict, List, Optional, Any
# import base64
# from io import BytesIO
# import torch
# from transformers import AutoProcessor, AutoModel

# from .base_agent import BaseAgent

# class TextToAudioAgent(BaseAgent):
#     """Agent for text-to-audio generation tasks."""
    
#     def __init__(self, 
#                 name: str = "Text-to-Audio Agent", 
#                 description: str = "Generates speech from text",
#                 agent_id: Optional[str] = None,
#                 model_name: str = "facebook/mms-tts-eng"):
#         """
#         Initialize the text-to-audio agent.
        
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
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.processor = AutoProcessor.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name).to(self.device)
    
#     def get_capabilities(self) -> List[str]:
#         """Return a list of capabilities this agent provides."""
#         return [
#             "text_to_audio",
#             "text_to_speech",
#             "voice_generation"
#         ]
    
#     async def process(self, task_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """
#         Process a text-to-audio task.
        
#         Args:
#             task_data: Contains 'text' and voice parameters
#             context: Optional additional context
            
#         Returns:
#             Dict containing the generated audio as base64
#         """
#         start_time = time.time()
#         success = False
#         result = {}
        
#         try:
#             # Extract parameters
#             text = task_data.get("text")
#             if not text:
#                 raise ValueError("No text provided")
            
#             # Generate the audio
#             inputs = self.processor(text=text, return_tensors="pt").to(self.device)
#             output = self.model.generate(**inputs)
            
#             # Convert to base64
#             audio_data = base64.b64encode(output.cpu().numpy().tobytes()).decode("utf-8")
            
#             result = {
#                 "audio_data": f"data:audio/wav;base64,{audio_data}",
#                 "text": text,
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