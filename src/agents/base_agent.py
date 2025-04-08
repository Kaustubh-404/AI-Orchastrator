from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import time


class BaseAgent(ABC):
    """Base interface for all specialized AI agents."""
    
    def __init__(self, agent_id: str, name: str, description: str):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.last_health_check = time.time()
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0,
        }
    
    @abstractmethod
    async def process(self, task_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a task and return results.
        
        Args:
            task_data: The data for the task to process
            context: Optional additional context
            
        Returns:
            Dict containing the results of the task
        """
        pass
    
    def get_capabilities(self) -> List[str]:
        """Return a list of capabilities this agent provides."""
        raise NotImplementedError("Subclasses must implement get_capabilities")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return performance metrics for this agent."""
        if self.metrics["total_requests"] > 0:
            avg_response_time = self.metrics["total_processing_time"] / self.metrics["total_requests"]
            success_rate = self.metrics["successful_requests"] / self.metrics["total_requests"] * 100
        else:
            avg_response_time = 0
            success_rate = 0
            
        return {
            "total_requests": self.metrics["total_requests"],
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
        }
    
    async def health_check(self) -> bool:
        """Verify agent is operational."""
        self.last_health_check = time.time()
        return True
    
    def update_metrics(self, success: bool, processing_time: float) -> None:
        """
        Update the agent's performance metrics.
        
        Args:
            success: Whether the request was successful
            processing_time: How long the request took to process in seconds
        """
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        self.metrics["total_processing_time"] += processing_time