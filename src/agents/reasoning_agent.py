import time
from typing import Dict, List, Optional, Any
import uuid
import os

from langchain_groq import ChatGroq
from .base_agent import BaseAgent


class ReasoningAgent(BaseAgent):
    """Agent for reasoning and problem-solving tasks."""
    
    def __init__(self, 
                 name: str = "Reasoning Agent", 
                 description: str = "Performs logical reasoning and problem-solving",
                 agent_id: Optional[str] = None):
        """
        Initialize the reasoning agent.
        
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
            "reasoning",
            "problem_solving",
            "logical_deduction",
            "mathematical_computation",
            "planning"
        ]
    
    async def process(self, task_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a reasoning task.
        
        Args:
            task_data: Contains 'problem' and other details
            context: Optional additional context
            
        Returns:
            Dict containing the reasoning result
        """
        start_time = time.time()
        success = False
        result = {}
        
        try:
            # Extract parameters
            problem = task_data.get("problem", task_data.get("description", ""))
            
            # Create a structured prompt to enhance reasoning abilities
            structured_prompt = f"""
            I need to solve the following problem using logical reasoning:
            
            PROBLEM: {problem}
            
            Let me think through this step by step:
            1. First, I'll identify the key information and variables.
            2. Next, I'll determine what approach or method is most appropriate.
            3. Then, I'll apply the approach methodically.
            4. Finally, I'll verify my solution and present the answer.
            
            REASONING:
            """
            
            # Generate reasoning
            messages = [{"role": "user", "content": structured_prompt}]
            response = self.llm.invoke(messages)
            reasoning_text = response.content
            
            # Extract a concise answer/conclusion from the reasoning
            conclusion_prompt = f"""
            Based on this reasoning:
            
            {reasoning_text}
            
            What is the final conclusion or answer to the original problem?
            
            CONCISE ANSWER:
            """
            
            messages = [{"role": "user", "content": conclusion_prompt}]
            conclusion_response = self.llm.invoke(messages)
            conclusion = conclusion_response.content
            
            result = {
                "task_id": task_data.get("task_id"),
                "status": "completed",
                "reasoning": reasoning_text,
                "conclusion": conclusion
            }
            success = True
            
        except Exception as e:
            result = {
                "task_id": task_data.get("task_id"),
                "status": "failed",
                "error": str(e)
            }
        
        # Update metrics
        processing_time = time.time() - start_time
        self.update_metrics(success, processing_time)
        
        return result