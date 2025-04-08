from typing import Dict, Any, Optional
import time
import asyncio
import uuid

from ..analyzer.task_analyzer import TaskAnalyzer
from ..registry.agent_registry import AgentRegistry
from ..execution.engine import ExecutionEngine
from ..integrator.output_integrator import OutputIntegrator


class Orchestrator:
    """
    Core orchestrator that coordinates the entire workflow.
    """
    
    def __init__(self, agent_registry: Optional[AgentRegistry] = None):
        """
        Initialize the orchestrator with its components.
        
        Args:
            agent_registry: Optional AgentRegistry to use
        """
        self.agent_registry = agent_registry or AgentRegistry()
        self.task_analyzer = TaskAnalyzer()
        self.execution_engine = ExecutionEngine(self.agent_registry)
        self.output_integrator = OutputIntegrator()
        
        # Track request history
        self.requests = {}
    
    async def process_request(self, request_content: str, request_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user request through the entire orchestration pipeline.
        
        Args:
            request_content: The content of the user request
            request_context: Optional additional context for the request
            
        Returns:
            Dict containing the final response and metadata
        """
        start_time = time.time()
        
        # Generate request ID if not provided
        request_id = str(uuid.uuid4())
        request_context = request_context or {}
        
        # Store request
        self.requests[request_id] = {
            "request_id": request_id,
            "content": request_content,
            "context": request_context,
            "status": "processing",
            "created_at": start_time,
            "stages": {}
        }
        
        try:
            # Step 1: Analyze the request
            analysis_start = time.time()
            analysis_result = await self.task_analyzer.analyze_request(request_content)
            analysis_time = time.time() - analysis_start
            
            # Store analysis result
            self.requests[request_id]["stages"]["analysis"] = {
                "result": analysis_result,
                "time": analysis_time
            }
            
            # Step 2: Execute the request
            execution_start = time.time()
            execution_result = await self.execution_engine.execute_request(analysis_result)
            execution_time = time.time() - execution_start
            
            # Store execution result
            self.requests[request_id]["stages"]["execution"] = {
                "result": execution_result,
                "time": execution_time
            }
            
            # Step 3: Integrate the results
            integration_start = time.time()
            integration_result = await self.output_integrator.integrate_results(
                execution_result=execution_result,
                original_request=request_content
            )
            integration_time = time.time() - integration_start
            
            # Store integration result
            self.requests[request_id]["stages"]["integration"] = {
                "result": integration_result,
                "time": integration_time
            }
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Update request status
            self.requests[request_id]["status"] = "completed"
            self.requests[request_id]["completed_at"] = time.time()
            self.requests[request_id]["processing_time"] = total_time
            
            # Prepare final response
            final_response = {
                "request_id": request_id,
                "status": "completed",
                "response": integration_result.get("integrated_response"),
                "processing_time": total_time,
                "metadata": {
                    "analysis_time": analysis_time,
                    "execution_time": execution_time,
                    "integration_time": integration_time,
                    "num_subtasks": len(analysis_result.get("subtasks", [])),
                    "success_rate": execution_result.get("success_rate", 0)
                }
            }
            
            # Add detailed results if requested
            if request_context.get("include_details"):
                final_response["details"] = {
                    "analysis": analysis_result,
                    "execution": execution_result,
                    "integration": integration_result
                }
            
            return final_response
            
        except Exception as e:
            # Handle errors
            error_time = time.time() - start_time
            
            # Update request status
            self.requests[request_id]["status"] = "failed"
            self.requests[request_id]["completed_at"] = time.time()
            self.requests[request_id]["processing_time"] = error_time
            self.requests[request_id]["error"] = str(e)
            
            # Return error response
            return {
                "request_id": request_id,
                "status": "failed",
                "error": str(e),
                "processing_time": error_time
            }
    
    def register_agent(self, agent) -> None:
        """
        Register a new agent with the orchestrator.
        
        Args:
            agent: The agent to register
        """
        self.agent_registry.register_agent(agent)
    
    def get_request_history(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the history of requests.
        
        Args:
            request_id: Optional ID of a specific request to retrieve
            
        Returns:
            Dict containing request history
        """
        if request_id:
            return self.requests.get(request_id, {"error": "Request not found"})
        
        return {
            "total_requests": len(self.requests),
            "requests": [
                {
                    "request_id": req_id,
                    "content": req_data.get("content", "")[:100] + "...",  # Truncate for brevity
                    "status": req_data.get("status"),
                    "created_at": req_data.get("created_at"),
                    "completed_at": req_data.get("completed_at", None),
                    "processing_time": req_data.get("processing_time", None)
                }
                for req_id, req_data in self.requests.items()
            ]
        }