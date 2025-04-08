import time
from typing import Dict, List, Optional, Any
import uuid
import os

from langchain_groq import ChatGroq
from .base_agent import BaseAgent


class TextGenerationAgent(BaseAgent):
    """Agent for text generation tasks."""
    
    def __init__(self, 
                 name: str = "Text Generation Agent", 
                 description: str = "Generates text based on prompts",
                 agent_id: Optional[str] = None):
        """
        Initialize the text generation agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent
            agent_id: Unique ID for the agent (generated if not provided)
        """
        if agent_id is None:
            agent_id = str(uuid.uuid4())
        super().__init__(agent_id, name, description)
        
        # Initialize the Groq LLM
        self.llm = ChatGroq(
            model_name="llama3-8b-8192",  # You can change this to other available Groq models
            groq_api_key=os.environ.get("GROQ_API_KEY")
        )
    
    def get_capabilities(self) -> List[str]:
        """Return a list of capabilities this agent provides."""
        return [
            "text_generation",
            "content_creation",
            "summarization",
            "paraphrasing",
        ]
    
    async def process(self, task_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a text generation task.
        
        Args:
            task_data: Contains 'prompt' and optional parameters like max_tokens
            context: Optional additional context
            
        Returns:
            Dict containing the generated text
        """
        start_time = time.time()
        success = False
        result = {}
        
        try:
            # Extract parameters
            prompt = task_data.get("prompt", "")
            max_tokens = task_data.get("max_tokens", 500)
            
            # Generate text using Groq's chat model
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)
            generated_text = response.content
            
            result = {
                "generated_text": generated_text,
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