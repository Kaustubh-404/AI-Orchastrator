from typing import Dict, List, Any
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


class OutputIntegrator:
    """
    Component that combines outputs from multiple agents into a cohesive response.
    """
    
    def __init__(self, llm=None):
        """
        Initialize the output integrator.
        
        Args:
            llm: Optional LLM to use for integration
        """
        self.llm = llm or ChatGroq(
            model_name="llama3-8b-8192",
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.2
        )
        
        # Template for output integration
        self.integration_template = PromptTemplate(
            input_variables=["results", "original_request"],
            template="""
            You are an AI output integrator. Your job is to combine the results from multiple specialized AI agents 
            into a cohesive response for the user. 
            
            Original user request: {original_request}
            
            Results from various AI agents:
            {results}
            
            Create a well-structured, cohesive response that addresses the original request using all the information 
            provided by the specialized agents. The response should flow naturally and not appear as separate pieces.
            
            Final integrated response:
            """
        )
    
    async def integrate_results(self, execution_result: Dict[str, Any], original_request: str) -> Dict[str, Any]:
        """
        Integrate results from multiple agents into a cohesive response.
        
        Args:
            execution_result: Results from the execution engine
            original_request: The original user request
            
        Returns:
            Dict containing the integrated response
        """
        # Extract task results
        task_results = execution_result["task_results"]
        
        # Format results for the integration prompt
        formatted_results = ""
        for task_id, result in task_results.items():
            if result.get("status") == "completed":
                task_data = result.get("result", {})
                # Extract text content from task data
                if isinstance(task_data, dict):
                    content = task_data.get("generated_text", "")
                    if not content and "content" in task_data:
                        content = task_data["content"]
                    if not content:
                        # Try to get any string value from the dict
                        for key, value in task_data.items():
                            if isinstance(value, str) and value:
                                content = value
                                break
                else:
                    content = str(task_data)
                
                # Add to formatted results
                formatted_results += f"Task: {result.get('task_id')}\n"
                formatted_results += f"Result: {content}\n\n"
            else:
                # Add failed task info
                formatted_results += f"Task: {result.get('task_id')} (Failed)\n"
                formatted_results += f"Error: {result.get('error', 'Unknown error')}\n\n"
        
        # If there are no usable results, return a simple response
        if not formatted_results.strip():
            return {
                "request_id": execution_result.get("request_id"),
                "integrated_response": "I'm sorry, but I couldn't process your request successfully. Please try again or rephrase your request.",
                "status": "failed"
            }
        
        # Use the LLM to integrate the results
        try:
            integrated_prompt = self.integration_template.format(
                results=formatted_results,
                original_request=original_request
            )
            
            messages = [{"role": "user", "content": integrated_prompt}]
            response = self.llm.invoke(messages)
            integrated_response = response.content
            
            return {
                "request_id": execution_result.get("request_id"),
                "integrated_response": integrated_response,
                "raw_results": task_results,
                "status": "completed"
            }
            
        except Exception as e:
            # Fallback to a simple concatenation of results
            simple_response = "Here are the results of your request:\n\n" + formatted_results
            
            return {
                "request_id": execution_result.get("request_id"),
                "integrated_response": simple_response,
                "raw_results": task_results,
                "status": "partial",
                "error": str(e)
            }


    # Update src/integrator/output_integrator.py

# Add this method to the OutputIntegrator class

async def integrate_multimedia_results(self, execution_result: Dict[str, Any], original_request: str) -> Dict[str, Any]:
    """
    Integrate multimedia results from multiple agents.
    
    Args:
        execution_result: Results from the execution engine
        original_request: The original user request
        
    Returns:
        Dict containing the integrated response with media data
    """
    # Extract task results
    task_results = execution_result["task_results"]
    
    # Collect media data from different tasks
    text_content = ""
    image_data = []
    audio_data = None
    video_data = None
    
    for task_id, result in task_results.items():
        if result.get("status") == "completed":
            task_data = result.get("result", {})
            
            # Collect text content
            if "generated_text" in task_data:
                text_content += task_data["generated_text"] + "\n\n"
            elif "text_description" in task_data:
                text_content += task_data["text_description"] + "\n\n"
            elif "transcript" in task_data:
                text_content += task_data["transcript"] + "\n\n"
            elif "conclusion" in task_data:
                text_content += task_data["conclusion"] + "\n\n"
            
            # Collect image data
            if "image_data" in task_data:
                image_data.append(task_data["image_data"])
            
            # Collect audio data
            if "audio_data" in task_data:
                audio_data = task_data["audio_data"]
            
            # Collect video data (takes precedence)
            if "video_data" in task_data:
                video_data = task_data["video_data"]
                # If we have video, that's our final output
                break
    
    # Prepare the response
    response = {
        "request_id": execution_result.get("request_id"),
        "text_content": text_content,
        "raw_results": task_results,
        "status": "completed"
    }
    
    # Add media data if available
    if video_data:
        response["video_data"] = video_data
    elif audio_data and image_data:
        # If we have both audio and images but no video, they might need to be combined
        # This would be handled by a subsequent video creation task, but add them separately for now
        response["audio_data"] = audio_data
        response["image_data"] = image_data[0] if image_data else None
    elif image_data:
        response["image_data"] = image_data[0] if image_data else None
    elif audio_data:
        response["audio_data"] = audio_data
    
    return response