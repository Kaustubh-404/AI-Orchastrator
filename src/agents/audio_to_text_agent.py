# src/agents/audio_to_text_agent.py
import time
import uuid
import os
from typing import Dict, List, Optional, Any
import base64
from io import BytesIO
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from .base_agent import BaseAgent

class AudioToTextAgent(BaseAgent):
    """Agent for audio-to-text transcription tasks."""
    
    def __init__(self, 
                name: str = "Audio-to-Text Agent", 
                description: str = "Transcribes speech to text",
                agent_id: Optional[str] = None,
                model_name: str = "openai/whisper-large-v2"):
        """
        Initialize the audio-to-text agent.
        
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(self.device)
    
    def get_capabilities(self) -> List[str]:
        """Return a list of capabilities this agent provides."""
        return [
            "audio_to_text",
            "speech_recognition",
            "transcription"
        ]
    
    async def process(self, task_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an audio-to-text task.
        
        Args:
            task_data: Contains 'audio_data' (base64 or file path)
            context: Optional additional context
            
        Returns:
            Dict containing the transcribed text
        """
        start_time = time.time()
        success = False
        result = {}
        
        try:
            # Extract parameters
            audio_data = task_data.get("audio_data")
            if not audio_data:
                raise ValueError("No audio data provided")
            
            # Load and process the audio
            audio_bytes = self._load_audio(audio_data)
            inputs = self.processor(audio_bytes, return_tensors="pt").to(self.device)
            
            # Generate transcription
            output = self.model.generate(**inputs)
            transcript = self.processor.decode(output[0], skip_special_tokens=True)
            
            result = {
                "transcript": transcript,
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
    
    def _load_audio(self, audio_data):
        """
        Load audio from various formats.
        
        Args:
            audio_data: Base64 string, file path, or raw bytes
            
        Returns:
            Audio as numpy array
        """
        # Handle base64 data
        if isinstance(audio_data, str) and audio_data.startswith('data:audio'):
            # Extract the base64 part
            base64_data = audio_data.split(',')[1]
            audio_bytes = base64.b64decode(base64_data)
            return audio_bytes
        
        # Handle file path
        elif isinstance(audio_data, str) and os.path.exists(audio_data):
            with open(audio_data, 'rb') as f:
                return f.read()
        
        # Handle raw bytes
        elif isinstance(audio_data, bytes):
            return audio_data
        
        else:
            raise ValueError("Unsupported audio data format")