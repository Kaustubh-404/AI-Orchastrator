# from typing import Dict, Any, Optional
# import time
# import asyncio
# import uuid

# from ..analyzer.task_analyzer import TaskAnalyzer
# from ..registry.agent_registry import AgentRegistry
# from ..execution.engine import ExecutionEngine
# from ..integrator.output_integrator import OutputIntegrator


# class Orchestrator:
#     """
#     Core orchestrator that coordinates the entire workflow.
#     """
    
#     def __init__(self, agent_registry: Optional[AgentRegistry] = None):
#         """
#         Initialize the orchestrator with its components.
        
#         Args:
#             agent_registry: Optional AgentRegistry to use
#         """
#         self.agent_registry = agent_registry or AgentRegistry()
#         self.task_analyzer = TaskAnalyzer()
#         self.execution_engine = ExecutionEngine(self.agent_registry)
#         self.output_integrator = OutputIntegrator()
        
#         # Track request history
#         self.requests = {}

#     # Update src/core/orchestrator.py - modify the process_request method

#     async def process_request(self, request_content: str, request_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """
#         Process a user request through the entire orchestration pipeline.
        
#         Args:
#             request_content: The content of the user request
#             request_context: Optional additional context for the request
            
#         Returns:
#             Dict containing the final response and metadata
#         """
#         start_time = time.time()
        
#         # Generate request ID if not provided
#         request_id = str(uuid.uuid4())
#         request_context = request_context or {}
        
#         # Store request
#         self.requests[request_id] = {
#             "request_id": request_id,
#             "content": request_content,
#             "context": request_context,
#             "status": "processing",
#             "created_at": start_time,
#             "stages": {}
#         }
        
#         try:
#             # Step 1: Analyze the request
#             analysis_start = time.time()
#             analysis_result = await self.task_analyzer.analyze_request(request_content)
#             analysis_time = time.time() - analysis_start
            
#             # Store analysis result
#             self.requests[request_id]["stages"]["analysis"] = {
#                 "result": analysis_result,
#                 "time": analysis_time
#             }
            
#             # Step 2: Execute the request
#             execution_start = time.time()
#             execution_result = await self.execution_engine.execute_request(analysis_result)
#             execution_time = time.time() - execution_start
            
#             # Store execution result
#             self.requests[request_id]["stages"]["execution"] = {
#                 "result": execution_result,
#                 "time": execution_time
#             }
            
#             # Step 3: Integrate the results
#             integration_start = time.time()
            
#             # Check if this is a multimedia request
#             is_multimedia = False
#             for subtask in analysis_result.get("subtasks", []):
#                 capability = subtask.get("required_capability", "")
#                 if capability in ["text_to_image", "text_to_audio", "image_to_text", 
#                                 "audio_to_text", "video_creation"]:
#                     is_multimedia = True
#                     break
            
#             # Choose appropriate integration method
#             if is_multimedia:
#                 integration_result = await self.output_integrator.integrate_multimedia_results(
#                     execution_result=execution_result,
#                     original_request=request_content
#                 )
#             else:
#                 integration_result = await self.output_integrator.integrate_results(
#                     execution_result=execution_result,
#                     original_request=request_content
#                 )
                
#             integration_time = time.time() - integration_start
            
#             # Store integration result
#             self.requests[request_id]["stages"]["integration"] = {
#                 "result": integration_result,
#                 "time": integration_time
#             }
            
#             # Calculate total processing time
#             total_time = time.time() - start_time
            
#             # Update request status
#             self.requests[request_id]["status"] = "completed"
#             self.requests[request_id]["completed_at"] = time.time()
#             self.requests[request_id]["processing_time"] = total_time
            
#             # Prepare final response
#             final_response = {
#                 "request_id": request_id,
#                 "status": "completed",
#                 "processing_time": total_time,
#                 "metadata": {
#                     "analysis_time": analysis_time,
#                     "execution_time": execution_time,
#                     "integration_time": integration_time,
#                     "num_subtasks": len(analysis_result.get("subtasks", [])),
#                     "success_rate": execution_result.get("success_rate", 0)
#                 }
#             }
            
#             # Add response content based on type
#             if is_multimedia:
#                 # For multimedia responses, include all media data
#                 if "video_data" in integration_result:
#                     final_response["response"] = {
#                         "type": "video",
#                         "video_data": integration_result.get("video_data"),
#                         "text_content": integration_result.get("text_content", "")
#                     }
#                 elif "audio_data" in integration_result:
#                     final_response["response"] = {
#                         "type": "audio",
#                         "audio_data": integration_result.get("audio_data"),
#                         "text_content": integration_result.get("text_content", "")
#                     }
#                 elif "image_data" in integration_result:
#                     final_response["response"] = {
#                         "type": "image",
#                         "image_data": integration_result.get("image_data"),
#                         "text_content": integration_result.get("text_content", "")
#                     }
#                 else:
#                     final_response["response"] = integration_result.get("text_content", "")
#             else:
#                 # For text-only responses
#                 final_response["response"] = integration_result.get("integrated_response", "")
            
#             # Add detailed results if requested
#             if request_context.get("include_details"):
#                 final_response["details"] = {
#                     "analysis": analysis_result,
#                     "execution": execution_result,
#                     "integration": integration_result
#                 }
            
#             return final_response
            
#         except Exception as e:
#             # Handle errors
#             error_time = time.time() - start_time
            
#             # Update request status
#             self.requests[request_id]["status"] = "failed"
#             self.requests[request_id]["completed_at"] = time.time()
#             self.requests[request_id]["processing_time"] = error_time
#             self.requests[request_id]["error"] = str(e)
            
#             # Return error response
#             return {
#                 "request_id": request_id,
#                 "status": "failed",
#                 "error": str(e),
#                 "processing_time": error_time
#             }
    
#     # async def process_request(self, request_content: str, request_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     #     """
#     #     Process a user request through the entire orchestration pipeline.
        
#     #     Args:
#     #         request_content: The content of the user request
#     #         request_context: Optional additional context for the request
            
#     #     Returns:
#     #         Dict containing the final response and metadata
#     #     """
#     #     start_time = time.time()
        
#     #     # Generate request ID if not provided
#     #     request_id = str(uuid.uuid4())
#     #     request_context = request_context or {}
        
#     #     # Store request
#     #     self.requests[request_id] = {
#     #         "request_id": request_id,
#     #         "content": request_content,
#     #         "context": request_context,
#     #         "status": "processing",
#     #         "created_at": start_time,
#     #         "stages": {}
#     #     }
        
#     #     try:
#     #         # Step 1: Analyze the request
#     #         analysis_start = time.time()
#     #         analysis_result = await self.task_analyzer.analyze_request(request_content)
#     #         analysis_time = time.time() - analysis_start
            
#     #         # Store analysis result
#     #         self.requests[request_id]["stages"]["analysis"] = {
#     #             "result": analysis_result,
#     #             "time": analysis_time
#     #         }
            
#     #         # Step 2: Execute the request
#     #         execution_start = time.time()
#     #         execution_result = await self.execution_engine.execute_request(analysis_result)
#     #         execution_time = time.time() - execution_start
            
#     #         # Store execution result
#     #         self.requests[request_id]["stages"]["execution"] = {
#     #             "result": execution_result,
#     #             "time": execution_time
#     #         }
            
#     #         # Step 3: Integrate the results
#     #         integration_start = time.time()
#     #         integration_result = await self.output_integrator.integrate_results(
#     #             execution_result=execution_result,
#     #             original_request=request_content
#     #         )
#     #         integration_time = time.time() - integration_start
            
#     #         # Store integration result
#     #         self.requests[request_id]["stages"]["integration"] = {
#     #             "result": integration_result,
#     #             "time": integration_time
#     #         }
            
#     #         # Calculate total processing time
#     #         total_time = time.time() - start_time
            
#     #         # Update request status
#     #         self.requests[request_id]["status"] = "completed"
#     #         self.requests[request_id]["completed_at"] = time.time()
#     #         self.requests[request_id]["processing_time"] = total_time
            
#     #         # Prepare final response
#     #         final_response = {
#     #             "request_id": request_id,
#     #             "status": "completed",
#     #             "response": integration_result.get("integrated_response"),
#     #             "processing_time": total_time,
#     #             "metadata": {
#     #                 "analysis_time": analysis_time,
#     #                 "execution_time": execution_time,
#     #                 "integration_time": integration_time,
#     #                 "num_subtasks": len(analysis_result.get("subtasks", [])),
#     #                 "success_rate": execution_result.get("success_rate", 0)
#     #             }
#     #         }
            
#     #         # Add detailed results if requested
#     #         if request_context.get("include_details"):
#     #             final_response["details"] = {
#     #                 "analysis": analysis_result,
#     #                 "execution": execution_result,
#     #                 "integration": integration_result
#     #             }
            
#     #         return final_response
            
#     #     except Exception as e:
#     #         # Handle errors
#     #         error_time = time.time() - start_time
            
#     #         # Update request status
#     #         self.requests[request_id]["status"] = "failed"
#     #         self.requests[request_id]["completed_at"] = time.time()
#     #         self.requests[request_id]["processing_time"] = error_time
#     #         self.requests[request_id]["error"] = str(e)
            
#     #         # Return error response
#     #         return {
#     #             "request_id": request_id,
#     #             "status": "failed",
#     #             "error": str(e),
#     #             "processing_time": error_time
#     #         }
    
#     def register_agent(self, agent) -> None:
#         """
#         Register a new agent with the orchestrator.
        
#         Args:
#             agent: The agent to register
#         """
#         self.agent_registry.register_agent(agent)
    
#     def get_request_history(self, request_id: Optional[str] = None) -> Dict[str, Any]:
#         """
#         Get the history of requests.
        
#         Args:
#             request_id: Optional ID of a specific request to retrieve
            
#         Returns:
#             Dict containing request history
#         """
#         if request_id:
#             return self.requests.get(request_id, {"error": "Request not found"})
        
#         return {
#             "total_requests": len(self.requests),
#             "requests": [
#                 {
#                     "request_id": req_id,
#                     "content": req_data.get("content", "")[:100] + "...",  # Truncate for brevity
#                     "status": req_data.get("status"),
#                     "created_at": req_data.get("created_at"),
#                     "completed_at": req_data.get("completed_at", None),
#                     "processing_time": req_data.get("processing_time", None)
#                 }
#                 for req_id, req_data in self.requests.items()
#             ]
#         }



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

    # In src/core/orchestrator.py, update the process_request method:

    async def process_request(self, request_content: str, request_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user request through the entire orchestration pipeline."""
        start_time = time.time()
        
        # Use the request_id from context if provided, otherwise generate a new one
        request_context = request_context or {}
        request_id = request_context.get("request_id", str(uuid.uuid4()))
        
        print(f"Starting to process request {request_id}")
        
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
            print(f"Analyzing request {request_id}")
            analysis_start = time.time()
            analysis_result = await self.task_analyzer.analyze_request(request_content, request_id)
            analysis_time = time.time() - analysis_start
            
            # Override the request_id in the analysis result to match our expected one
            analysis_result["request_id"] = request_id
            
            print(f"Analysis completed for request {request_id}. Found {len(analysis_result.get('subtasks', []))} subtasks.")
            
            # Store analysis result
            self.requests[request_id]["stages"]["analysis"] = {
                "result": analysis_result,
                "time": analysis_time
            }
            
            # Step 2: Execute the request
            print(f"Executing subtasks for request {request_id}")
            execution_start = time.time()
            execution_result = await self.execution_engine.execute_request(analysis_result)
            execution_time = time.time() - execution_start
            
            # Ensure the execution result has the correct request_id
            execution_result["request_id"] = request_id
            
            print(f"Execution completed for request {request_id}. Status: {execution_result.get('overall_status', 'unknown')}")
            
            # Store execution result
            self.requests[request_id]["stages"]["execution"] = {
                "result": execution_result,
                "time": execution_time
            }
            
            # Step 3: Integrate the results
            print(f"Integrating results for request {request_id}")
            integration_start = time.time()
            
            # Check if this is a multimedia request
            is_multimedia = False
            media_type = None
            for subtask in analysis_result.get("subtasks", []):
                capability = subtask.get("required_capability", "")
                if capability == "text_to_image":
                    is_multimedia = True
                    media_type = "image"
                    print(f"Request {request_id} identified as multimedia request due to capability: {capability}")
                    break
                elif capability == "text_to_audio":
                    is_multimedia = True
                    media_type = "audio"
                    print(f"Request {request_id} identified as multimedia request due to capability: {capability}")
                    break
                elif capability == "video_creation":
                    is_multimedia = True
                    media_type = "video"
                    print(f"Request {request_id} identified as multimedia request due to capability: {capability}")
                    break
            
            # Choose appropriate integration method
            if is_multimedia:
                print(f"Using multimedia integration for request {request_id}")
                integration_result = await self.output_integrator.integrate_multimedia_results(
                    execution_result=execution_result,
                    original_request=request_content
                )
            else:
                print(f"Using standard integration for request {request_id}")
                integration_result = await self.output_integrator.integrate_results(
                    execution_result=execution_result,
                    original_request=request_content
                )
                
            # Ensure the integration result has the correct request_id
            integration_result["request_id"] = request_id
                
            integration_time = time.time() - integration_start
            
            print(f"Integration completed for request {request_id}")
            
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
            
            # Prepare final response based on results
            # We make sure to correctly structure multimedia responses
            final_response = {
                "request_id": request_id,
                "status": "completed",
                "processing_time": total_time,
                "metadata": {
                    "analysis_time": analysis_time,
                    "execution_time": execution_time,
                    "integration_time": integration_time,
                    "num_subtasks": len(analysis_result.get("subtasks", [])),
                    "success_rate": execution_result.get("success_rate", 0)
                }
            }
            
            # Collect media data from task results
            media_data = None
            for task_id, result in execution_result.get("task_results", {}).items():
                if result.get("status") == "completed" and isinstance(result.get("result"), dict):
                    task_data = result.get("result", {})
                    
                    # Check for media data
                    if media_type == "audio" and "audio_data" in task_data:
                        media_data = task_data["audio_data"]
                        print(f"Found audio data in task {task_id}")
                    elif media_type == "image" and "image_data" in task_data:
                        media_data = task_data["image_data"]
                        print(f"Found image data in task {task_id}")
                    elif media_type == "video" and "video_data" in task_data:
                        media_data = task_data["video_data"]
                        print(f"Found video data in task {task_id}")
            
            # Add response content based on type
            if is_multimedia:
                # Check if we have media data or if it's in the integration result
                response_content = {}
                
                # Add text content if available
                text_content = integration_result.get("text_content", "")
                if text_content:
                    response_content["text_content"] = text_content
                
                # Add media data from either the integration result or what we collected
                if media_type == "audio":
                    audio_data = integration_result.get("audio_data") or media_data
                    if audio_data:
                        response_content["audio_data"] = audio_data
                        print(f"Adding audio data to response")
                elif media_type == "image":
                    image_data = integration_result.get("image_data") or media_data
                    if image_data:
                        response_content["image_data"] = image_data
                        print(f"Adding image data to response")
                elif media_type == "video":
                    video_data = integration_result.get("video_data") or media_data
                    if video_data:
                        response_content["video_data"] = video_data
                        print(f"Adding video data to response")
                
                # Set the response
                if response_content:
                    final_response["response"] = response_content
                    print(f"Set multimedia response with keys: {list(response_content.keys())}")
                else:
                    final_response["response"] = {"text_content": text_content or "No content generated"}
                    print(f"No media data found, setting text-only response")
            else:
                # For text-only responses
                final_response["response"] = {"text_content": integration_result.get("integrated_response", "")}
                print(f"Set text-only response")
            
            # Add detailed results if requested
            if request_context.get("include_details"):
                final_response["details"] = {
                    "analysis": analysis_result,
                    "execution": execution_result,
                    "integration": integration_result
                }
            
            print(f"Request {request_id} completed successfully")
            return final_response
            
        except Exception as e:
            # Handle errors
            error_time = time.time() - start_time
            error_message = f"Error processing request {request_id}: {str(e)}"
            print(error_message)
            import traceback
            print(traceback.format_exc())
            
            # Update request status
            self.requests[request_id]["status"] = "failed"
            self.requests[request_id]["completed_at"] = time.time()
            self.requests[request_id]["processing_time"] = error_time
            self.requests[request_id]["error"] = error_message
            
            # Return error response
            return {
                "request_id": request_id,
                "status": "failed",
                "error": error_message,
                "processing_time": error_time
            }

    # async def process_request(self, request_content: str, request_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    #     """Process a user request through the entire orchestration pipeline."""
    #     start_time = time.time()
        
    #     # Generate request ID if not provided
    #     request_id = str(uuid.uuid4())
    #     request_context = request_context or {}
        
    #     print(f"Starting to process request {request_id}")
        
    #     # Store request
    #     self.requests[request_id] = {
    #         "request_id": request_id,
    #         "content": request_content,
    #         "context": request_context,
    #         "status": "processing",
    #         "created_at": start_time,
    #         "stages": {}
    #     }
        
    #     try:
    #         # Step 1: Analyze the request
    #         print(f"Analyzing request {request_id}")
    #         analysis_start = time.time()
    #         analysis_result = await self.task_analyzer.analyze_request(request_content)
    #         analysis_time = time.time() - analysis_start
            
    #         print(f"Analysis completed for request {request_id}. Found {len(analysis_result.get('subtasks', []))} subtasks.")
            
    #         # Store analysis result
    #         self.requests[request_id]["stages"]["analysis"] = {
    #             "result": analysis_result,
    #             "time": analysis_time
    #         }
            
    #         # Step 2: Execute the request
    #         print(f"Executing subtasks for request {request_id}")
    #         execution_start = time.time()
    #         execution_result = await self.execution_engine.execute_request(analysis_result)
    #         execution_time = time.time() - execution_start
            
    #         print(f"Execution completed for request {request_id}. Status: {execution_result.get('overall_status', 'unknown')}")
            
    #         # Store execution result
    #         self.requests[request_id]["stages"]["execution"] = {
    #             "result": execution_result,
    #             "time": execution_time
    #         }
            
    #         # Step 3: Integrate the results
    #         print(f"Integrating results for request {request_id}")
    #         integration_start = time.time()
            
    #         # Check if this is a multimedia request
    #         is_multimedia = False
    #         for subtask in analysis_result.get("subtasks", []):
    #             capability = subtask.get("required_capability", "")
    #             if capability in ["text_to_image", "text_to_audio", "image_to_text", 
    #                             "audio_to_text", "video_creation"]:
    #                 is_multimedia = True
    #                 print(f"Request {request_id} identified as multimedia request due to capability: {capability}")
    #                 break
            
    #         # Choose appropriate integration method
    #         if is_multimedia:
    #             print(f"Using multimedia integration for request {request_id}")
    #             integration_result = await self.output_integrator.integrate_multimedia_results(
    #                 execution_result=execution_result,
    #                 original_request=request_content
    #             )
    #         else:
    #             print(f"Using standard integration for request {request_id}")
    #             integration_result = await self.output_integrator.integrate_results(
    #                 execution_result=execution_result,
    #                 original_request=request_content
    #             )
                
    #         integration_time = time.time() - integration_start
            
    #         print(f"Integration completed for request {request_id}")
            
    #         # Store integration result
    #         self.requests[request_id]["stages"]["integration"] = {
    #             "result": integration_result,
    #             "time": integration_time
    #         }
            
    #         # Calculate total processing time
    #         total_time = time.time() - start_time
            
    #         # Update request status
    #         self.requests[request_id]["status"] = "completed"
    #         self.requests[request_id]["completed_at"] = time.time()
    #         self.requests[request_id]["processing_time"] = total_time
            
    #         # Prepare final response
    #         final_response = {
    #             "request_id": request_id,
    #             "status": "completed",
    #             "processing_time": total_time,
    #             "metadata": {
    #                 "analysis_time": analysis_time,
    #                 "execution_time": execution_time,
    #                 "integration_time": integration_time,
    #                 "num_subtasks": len(analysis_result.get("subtasks", [])),
    #                 "success_rate": execution_result.get("success_rate", 0)
    #             }
    #         }
            
    #         # Add response content based on type
    #         if is_multimedia:
    #             # For multimedia responses, include all media data
    #             if "video_data" in integration_result:
    #                 final_response["response"] = {
    #                     "type": "video",
    #                     "video_data": integration_result.get("video_data"),
    #                     "text_content": integration_result.get("text_content", "")
    #                 }
    #             elif "audio_data" in integration_result:
    #                 final_response["response"] = {
    #                     "type": "audio",
    #                     "audio_data": integration_result.get("audio_data"),
    #                     "text_content": integration_result.get("text_content", "")
    #                 }
    #             elif "image_data" in integration_result:
    #                 print(f"Adding image data to final response for request {request_id}")
    #                 final_response["response"] = {
    #                     "type": "image",
    #                     "image_data": integration_result.get("image_data"),
    #                     "text_content": integration_result.get("text_content", "")
    #                 }
    #             else:
    #                 final_response["response"] = integration_result.get("text_content", "")
    #         else:
    #             # For text-only responses
    #             final_response["response"] = integration_result.get("integrated_response", "")
            
    #         # Add detailed results if requested
    #         if request_context.get("include_details"):
    #             final_response["details"] = {
    #                 "analysis": analysis_result,
    #                 "execution": execution_result,
    #                 "integration": integration_result
    #             }
            
    #         print(f"Request {request_id} completed successfully")
    #         return final_response
            
    #     except Exception as e:
    #         # Handle errors
    #         error_time = time.time() - start_time
    #         error_message = f"Error processing request {request_id}: {str(e)}"
    #         print(error_message)
    #         import traceback
    #         print(traceback.format_exc())
            
    #         # Update request status
    #         self.requests[request_id]["status"] = "failed"
    #         self.requests[request_id]["completed_at"] = time.time()
    #         self.requests[request_id]["processing_time"] = error_time
    #         self.requests[request_id]["error"] = error_message
            
    #         # Return error response
    #         return {
    #             "request_id": request_id,
    #             "status": "failed",
    #             "error": error_message,
    #             "processing_time": error_time
    #         }

    # async def process_request(self, request_content: str, request_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    #     """
    #     Process a user request through the entire orchestration pipeline.
        
    #     Args:
    #         request_content: The content of the user request
    #         request_context: Optional additional context for the request
            
    #     Returns:
    #         Dict containing the final response and metadata
    #     """
    #     start_time = time.time()
        
    #     # Generate request ID if not provided
    #     request_id = str(uuid.uuid4())
    #     request_context = request_context or {}
        
    #     # Store request
    #     self.requests[request_id] = {
    #         "request_id": request_id,
    #         "content": request_content,
    #         "context": request_context,
    #         "status": "processing",
    #         "created_at": start_time,
    #         "stages": {}
    #     }
        
    #     try:
    #         # Step 1: Analyze the request
    #         analysis_start = time.time()
    #         analysis_result = await self.task_analyzer.analyze_request(request_content)
    #         analysis_time = time.time() - analysis_start
            
    #         # Store analysis result
    #         self.requests[request_id]["stages"]["analysis"] = {
    #             "result": analysis_result,
    #             "time": analysis_time
    #         }
            
    #         # Step 2: Execute the request
    #         execution_start = time.time()
    #         execution_result = await self.execution_engine.execute_request(analysis_result)
    #         execution_time = time.time() - execution_start
            
    #         # Store execution result
    #         self.requests[request_id]["stages"]["execution"] = {
    #             "result": execution_result,
    #             "time": execution_time
    #         }
            
    #         # Step 3: Integrate the results
    #         integration_start = time.time()
            
    #         # Check if this is a multimedia request
    #         is_multimedia = False
    #         for subtask in analysis_result.get("subtasks", []):
    #             capability = subtask.get("required_capability", "")
    #             if capability in ["text_to_image", "text_to_audio", "image_to_text", 
    #                             "audio_to_text", "video_creation"]:
    #                 is_multimedia = True
    #                 break
            
    #         # Choose appropriate integration method
    #         if is_multimedia:
    #             integration_result = await self.output_integrator.integrate_multimedia_results(
    #                 execution_result=execution_result,
    #                 original_request=request_content
    #             )
    #         else:
    #             integration_result = await self.output_integrator.integrate_results(
    #                 execution_result=execution_result,
    #                 original_request=request_content
    #             )
                
    #         integration_time = time.time() - integration_start
            
    #         # Store integration result
    #         self.requests[request_id]["stages"]["integration"] = {
    #             "result": integration_result,
    #             "time": integration_time
    #         }
            
    #         # Calculate total processing time
    #         total_time = time.time() - start_time
            
    #         # Update request status
    #         self.requests[request_id]["status"] = "completed"
    #         self.requests[request_id]["completed_at"] = time.time()
    #         self.requests[request_id]["processing_time"] = total_time
            
    #         # Prepare final response
    #         final_response = {
    #             "request_id": request_id,
    #             "status": "completed",
    #             "processing_time": total_time,
    #             "metadata": {
    #                 "analysis_time": analysis_time,
    #                 "execution_time": execution_time,
    #                 "integration_time": integration_time,
    #                 "num_subtasks": len(analysis_result.get("subtasks", [])),
    #                 "success_rate": execution_result.get("success_rate", 0)
    #             }
    #         }
            
    #         # Add response content based on type
    #         if is_multimedia:
    #             # For multimedia responses, include all media data
    #             if "video_data" in integration_result:
    #                 final_response["response"] = {
    #                     "type": "video",
    #                     "video_data": integration_result.get("video_data"),
    #                     "text_content": integration_result.get("text_content", "")
    #                 }
    #             elif "audio_data" in integration_result:
    #                 final_response["response"] = {
    #                     "type": "audio",
    #                     "audio_data": integration_result.get("audio_data"),
    #                     "text_content": integration_result.get("text_content", "")
    #                 }
    #             elif "image_data" in integration_result:
    #                 final_response["response"] = {
    #                     "type": "image",
    #                     "image_data": integration_result.get("image_data"),
    #                     "text_content": integration_result.get("text_content", "")
    #                 }
    #             else:
    #                 final_response["response"] = integration_result.get("text_content", "")
    #         else:
    #             # For text-only responses
    #             final_response["response"] = integration_result.get("integrated_response", "")
            
    #         # Add detailed results if requested
    #         if request_context.get("include_details"):
    #             final_response["details"] = {
    #                 "analysis": analysis_result,
    #                 "execution": execution_result,
    #                 "integration": integration_result
    #             }
            
    #         return final_response
            
    #     except Exception as e:
    #         # Handle errors
    #         error_time = time.time() - start_time
            
    #         # Update request status
    #         self.requests[request_id]["status"] = "failed"
    #         self.requests[request_id]["completed_at"] = time.time()
    #         self.requests[request_id]["processing_time"] = error_time
    #         self.requests[request_id]["error"] = str(e)
            
    #         # Return error response
    #         return {
    #             "request_id": request_id,
    #             "status": "failed",
    #             "error": str(e),
    #             "processing_time": error_time
    #         }
    
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