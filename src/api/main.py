import os
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import json
import uuid
from src.core.orchestrator import Orchestrator
from src.registry.agent_registry import AgentRegistry
from src.agents.text_agent import TextGenerationAgent
from src.agents.reasoning_agent import ReasoningAgent
from .models import RequestInput, RequestStatus, AgentInfo
from .routes import router as api_router

# Create the FastAPI app
app = FastAPI(
    title="AI Orchestrator",
    description="API for orchestrating multiple specialized AI agents",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create global orchestrator instance
agent_registry = AgentRegistry()
orchestrator = Orchestrator(agent_registry)

# In-memory storage for async operations
request_store = {}


# # # Initialize agents on startup
# # @app.on_event("startup")
# # async def startup_event():
#     # Register default agents
#     text_agent = TextGenerationAgent()
#     reasoning_agent = ReasoningAgent()
#     orchestrator.register_agent(text_agent)
#     orchestrator.register_agent(reasoning_agent)
    
#     # Start health checks
#     agent_registry.start_health_checks()

# Update the startup_event function in src/api/main.py

# @app.on_event("startup")
# async def startup_event():
#     # Register default agents
#     text_agent = TextGenerationAgent()
#     reasoning_agent = ReasoningAgent()
    
#     # Register multimedia agents
#     from src.agents.image_to_text_agent import ImageToTextAgent
#     from src.agents.text_to_image_agent import TextToImageAgent
#     from src.agents.text_to_audio_agent import TextToAudioAgent
#     from src.agents.audio_to_text_agent import AudioToTextAgent
#     from src.agents.video_creation_agent import VideoCreationAgent
    
#     image_to_text_agent = ImageToTextAgent()
#     text_to_image_agent = TextToImageAgent()
#     text_to_audio_agent = TextToAudioAgent()
#     audio_to_text_agent = AudioToTextAgent()
#     video_creation_agent = VideoCreationAgent()
    
#     # Register all agents with the orchestrator
#     orchestrator.register_agent(text_agent)
#     orchestrator.register_agent(reasoning_agent)
#     orchestrator.register_agent(image_to_text_agent)
#     orchestrator.register_agent(text_to_image_agent)
#     orchestrator.register_agent(text_to_audio_agent)
#     orchestrator.register_agent(audio_to_text_agent)
#     orchestrator.register_agent(video_creation_agent)
    
#     # Start health checks
#     agent_registry.start_health_checks()

@app.on_event("startup")
async def startup_event():
    # Register default agents
    text_agent = TextGenerationAgent()
    reasoning_agent = ReasoningAgent()
    
    # Register multimedia agents - if you've added these
    from src.agents.image_to_text_agent import ImageToTextAgent
    from src.agents.text_to_image_agent import TextToImageAgent
    from src.agents.text_to_audio_agent import TextToAudioAgent
    from src.agents.audio_to_text_agent import AudioToTextAgent
    from src.agents.video_creation_agent import VideoCreationAgent
    
    image_to_text_agent = ImageToTextAgent()
    text_to_image_agent = TextToImageAgent()
    text_to_audio_agent = TextToAudioAgent() 
    audio_to_text_agent = AudioToTextAgent()
    video_creation_agent = VideoCreationAgent()
    
    # Register all agents with the agent registry
    agent_registry.register_agent(text_agent)
    agent_registry.register_agent(reasoning_agent)
    agent_registry.register_agent(image_to_text_agent)
    agent_registry.register_agent(text_to_image_agent)
    agent_registry.register_agent(text_to_audio_agent)
    agent_registry.register_agent(audio_to_text_agent)
    agent_registry.register_agent(video_creation_agent)
    
    # Start health checks
    agent_registry.start_health_checks()

# Include API router
app.include_router(api_router)


# This is a modified version of the process_request_background function
# to be integrated into your src/api/main.py file

async def process_request_background(request_id: str, content: str, context: Optional[Dict[str, Any]] = None):
    """
    Process a request in the background and save any media results to the data directory.
    """
    import logging
    import base64
    import os
    from ..utils.media_storage import save_media_response, ensure_data_directory
    import json
    
    logger = logging.getLogger("api.request_processor")
    
    # Make sure data directory exists
    data_dir = ensure_data_directory()
    logger.info(f"Using data directory: {data_dir}")
    
    try:
        # Process the request with the orchestrator
        logger.info(f"Processing request {request_id}: {content[:100]}...")
        
        # IMPORTANT: Set the request_id in the context to ensure it's used throughout the pipeline
        if context is None:
            context = {}
        context["request_id"] = request_id
        
        result = await orchestrator.process_request(content, context)
        
        # Make sure the result contains the correct request_id
        if "request_id" in result and result["request_id"] != request_id:
            logger.warning(f"Request ID mismatch: {result['request_id']} vs {request_id}. Fixing...")
            result["request_id"] = request_id
        
        # Log the result structure
        result_keys = list(result.keys())
        response_type = type(result.get("response", ""))
        logger.info(f"Result keys: {result_keys}, Response type: {response_type}")
        
        # Save all of the result data for debugging
        debug_path = os.path.join(data_dir, f"full_result_{request_id}.json")
        with open(debug_path, 'w') as f:
            try:
                # Try to save as JSON, but handle non-serializable types
                json.dump(
                    {k: (v if not isinstance(v, (bytes, bytearray)) else "[binary data]") 
                     for k, v in result.items() if k != "response"},
                    f, 
                    indent=2, 
                    default=str
                )
                logger.info(f"Saved full result data to {debug_path}")
            except Exception as e:
                logger.error(f"Failed to save result data: {str(e)}")
        
        # Save any media in the response to the data directory
        response_data = result.get("response", {})
        
        # Make sure response_data is a dictionary
        if not isinstance(response_data, dict):
            logger.warning(f"Response is not a dictionary, converting: {type(response_data)}")
            if isinstance(response_data, str):
                response_data = {"text_content": response_data}
            else:
                response_data = {"text_content": str(response_data)}
            # Update the result
            result["response"] = response_data
        
        # Check for media data directly in the result or response
        for location in [result, response_data]:
            if "audio_data" in location:
                # Save the audio data
                audio_path = os.path.join(data_dir, f"audio_{request_id}.wav")
                try:
                    audio_data_str = location["audio_data"]
                    if ',' in audio_data_str:
                        audio_data = audio_data_str.split(",")[1]
                    else:
                        audio_data = audio_data_str
                    
                    with open(audio_path, 'wb') as f:
                        f.write(base64.b64decode(audio_data))
                    logger.info(f"Saved audio data to {audio_path}")
                    
                    # Add to response if not already there
                    if "audio_data" not in response_data:
                        response_data["audio_data"] = location["audio_data"]
                    
                    # Add to media files list
                    if "media_files" not in request_store[request_id]:
                        request_store[request_id]["media_files"] = []
                    request_store[request_id]["media_files"].append(audio_path)
                except Exception as e:
                    logger.error(f"Failed to save audio data: {str(e)}")
        
        # Now use the media_storage utility
        media_result = save_media_response({"response": response_data}, f"request_{request_id}")
        
        # Log the results of media saving
        if media_result["saved_files"]:
            for file_info in media_result["saved_files"]:
                logger.info(f"Saved {file_info['type']} to {file_info['path']}")
                # Add to media files list
                if "media_files" not in request_store[request_id]:
                    request_store[request_id]["media_files"] = []
                request_store[request_id]["media_files"].append(file_info["path"])
        
        if media_result["errors"]:
            for error_info in media_result["errors"]:
                logger.error(f"Failed to save {error_info['type']}: {error_info['error']}")
        
        # Update the request store with the final response
        request_store[request_id] = {
            "request_id": request_id,
            "status": result.get("status", "completed"),
            "response": response_data,  # Always store as dictionary
            "processing_time": result.get("processing_time", 0),
            "created_at": request_store[request_id].get("created_at"),
            "completed_at": asyncio.get_event_loop().time(),
            "media_files": request_store[request_id].get("media_files", [])
        }
        
        # Log the final response structure
        logger.info(f"Final response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'non-dict'}")
        
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}", exc_info=True)
        # Handle errors
        request_store[request_id] = {
            "request_id": request_id,
            "status": "failed",
            "error": str(e),
            "created_at": request_store[request_id].get("created_at"),
            "completed_at": asyncio.get_event_loop().time(),
        }

# async def process_request_background(request_id: str, content: str, context: Optional[Dict[str, Any]] = None):
#     """
#     Process a request in the background and save any media results to the data directory.
#     """
#     import logging
#     from ..utils.media_storage import save_media_response, ensure_data_directory
    
#     logger = logging.getLogger("api.request_processor")
    
#     # Make sure data directory exists
#     data_dir = ensure_data_directory()
#     logger.info(f"Using data directory: {data_dir}")
    
#     try:
#         # Process the request with the orchestrator
#         logger.info(f"Processing request {request_id}: {content[:100]}...")
        
#         # IMPORTANT: Set the request_id in the context to ensure it's used throughout the pipeline
#         if context is None:
#             context = {}
#         context["request_id"] = request_id
        
#         result = await orchestrator.process_request(content, context)
        
#         # Make sure the result contains the correct request_id
#         if "request_id" in result and result["request_id"] != request_id:
#             logger.warning(f"Request ID mismatch: {result['request_id']} vs {request_id}. Fixing...")
#             result["request_id"] = request_id
        
#         # Log the result structure
#         result_keys = list(result.keys())
#         response_type = type(result.get("response", ""))
#         logger.info(f"Result keys: {result_keys}, Response type: {response_type}")
        
#         # Save any media in the response to the data directory
#         media_result = save_media_response(result, f"request_{request_id}")
        
#         # Log the results of media saving
#         if media_result["saved_files"]:
#             for file_info in media_result["saved_files"]:
#                 logger.info(f"Saved {file_info['type']} to {file_info['path']}")
        
#         if media_result["errors"]:
#             for error_info in media_result["errors"]:
#                 logger.error(f"Failed to save {error_info['type']}: {error_info['error']}")
        
#         # Format the response properly for storage
#         response_to_store = result.get("response", "")
        
#         # Ensure the response is properly structured for the API 
#         # If it's a string but we intended to have media, convert to dict
#         if isinstance(response_to_store, str) and (
#             "audio_data" in result or 
#             "video_data" in result or 
#             "image_data" in result or
#             len(media_result["saved_files"]) > 0
#         ):
#             logger.info("Converting string response to dictionary with text_content")
#             response_to_store = {
#                 "text_content": response_to_store
#             }
            
#             # Add any media data from the result directly to the response
#             if "audio_data" in result:
#                 response_to_store["audio_data"] = result["audio_data"]
#                 logger.info("Added audio_data to response from result")
#             if "image_data" in result:
#                 response_to_store["image_data"] = result["image_data"]
#                 logger.info("Added image_data to response from result")
#             if "video_data" in result:
#                 response_to_store["video_data"] = result["video_data"]
#                 logger.info("Added video_data to response from result")
            
#             # Also add any media from saved files if not already in the response
#             for file_info in media_result["saved_files"]:
#                 media_type = file_info["type"]
#                 if media_type == "audio" and "audio_data" not in response_to_store:
#                     with open(file_info["path"], "rb") as f:
#                         audio_bytes = f.read()
#                         audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
#                         response_to_store["audio_data"] = f"data:audio/wav;base64,{audio_b64}"
#                         logger.info("Added audio_data to response from saved file")
#                 elif media_type == "image" and "image_data" not in response_to_store:
#                     with open(file_info["path"], "rb") as f:
#                         image_bytes = f.read()
#                         image_b64 = base64.b64encode(image_bytes).decode("utf-8")
#                         response_to_store["image_data"] = f"data:image/png;base64,{image_b64}"
#                         logger.info("Added image_data to response from saved file")
#                 elif media_type == "video" and "video_data" not in response_to_store:
#                     with open(file_info["path"], "rb") as f:
#                         video_bytes = f.read()
#                         video_b64 = base64.b64encode(video_bytes).decode("utf-8")
#                         response_to_store["video_data"] = f"data:video/mp4;base64,{video_b64}"
#                         logger.info("Added video_data to response from saved file")
        
#         # Update the request store
#         request_store[request_id] = {
#             "request_id": request_id,
#             "status": result.get("status", "completed"),
#             "response": response_to_store,
#             "processing_time": result.get("processing_time", 0),
#             "created_at": request_store[request_id].get("created_at"),
#             "completed_at": asyncio.get_event_loop().time(),
#             "media_files": [f["path"] for f in media_result["saved_files"]] if media_result["saved_files"] else []
#         }
        
#         # Log the final response structure
#         if isinstance(response_to_store, dict):
#             logger.info(f"Final response keys: {list(response_to_store.keys())}")
#         else:
#             logger.info(f"Final response type: {type(response_to_store)}")
        
#     except Exception as e:
#         logger.error(f"Error processing request {request_id}: {str(e)}", exc_info=True)
#         # Handle errors
#         request_store[request_id] = {
#             "request_id": request_id,
#             "status": "failed",
#             "error": str(e),
#             "created_at": request_store[request_id].get("created_at"),
#             "completed_at": asyncio.get_event_loop().time(),
#         }

# async def process_request_background(request_id: str, content: str, context: Optional[Dict[str, Any]] = None):
#     """
#     Process a request in the background and save any media results to the data directory.
#     """
#     import logging
#     from ..utils.media_storage import save_media_response, ensure_data_directory
    
#     logger = logging.getLogger("api.request_processor")
    
#     # Make sure data directory exists
#     data_dir = ensure_data_directory()
#     logger.info(f"Using data directory: {data_dir}")
    
#     try:
#         # Process the request with the orchestrator
#         logger.info(f"Processing request {request_id}: {content[:100]}...")
        
#         # IMPORTANT: Set the request_id in the context to ensure it's used throughout the pipeline
#         if context is None:
#             context = {}
#         context["request_id"] = request_id
        
#         result = await orchestrator.process_request(content, context)
        
#         # Make sure the result contains the correct request_id
#         if "request_id" in result and result["request_id"] != request_id:
#             logger.warning(f"Request ID mismatch: {result['request_id']} vs {request_id}. Fixing...")
#             result["request_id"] = request_id
        
#         # Save any media in the response to the data directory
#         media_result = save_media_response(result, f"request_{request_id}")
        
#         # Log the results of media saving
#         if media_result["saved_files"]:
#             for file_info in media_result["saved_files"]:
#                 logger.info(f"Saved {file_info['type']} to {file_info['path']}")
        
#         if media_result["errors"]:
#             for error_info in media_result["errors"]:
#                 logger.error(f"Failed to save {error_info['type']}: {error_info['error']}")
        
#         # Update the request store with the correct request_id
#         request_store[request_id] = {
#             "request_id": request_id,
#             "status": result.get("status", "completed"),
#             "response": result.get("response", ""),
#             "processing_time": result.get("processing_time", 0),
#             "created_at": request_store[request_id].get("created_at"),
#             "completed_at": asyncio.get_event_loop().time(),
#             "media_files": [f["path"] for f in media_result["saved_files"]] if media_result["saved_files"] else []
#         }
        
#     except Exception as e:
#         logger.error(f"Error processing request {request_id}: {str(e)}", exc_info=True)
#         # Handle errors
#         request_store[request_id] = {
#             "request_id": request_id,
#             "status": "failed",
#             "error": str(e),
#             "created_at": request_store[request_id].get("created_at"),
#             "completed_at": asyncio.get_event_loop().time(),
#         }

# async def process_request_background(request_id: str, content: str, context: Optional[Dict[str, Any]] = None):
#     """
#     Process a request in the background and save any media results to the data directory.
#     """
#     import logging
#     from ..utils.media_storage import save_media_response, ensure_data_directory
    
#     logger = logging.getLogger("api.request_processor")
    
#     # Make sure data directory exists
#     data_dir = ensure_data_directory()
#     logger.info(f"Using data directory: {data_dir}")
    
#     try:
#         # Process the request with the orchestrator
#         logger.info(f"Processing request {request_id}: {content[:100]}...")
#         result = await orchestrator.process_request(content, context)
        
#         # Save any media in the response to the data directory
#         media_result = save_media_response(result, f"request_{request_id}")
        
#         # Log the results of media saving
#         if media_result["saved_files"]:
#             for file_info in media_result["saved_files"]:
#                 logger.info(f"Saved {file_info['type']} to {file_info['path']}")
        
#         if media_result["errors"]:
#             for error_info in media_result["errors"]:
#                 logger.error(f"Failed to save {error_info['type']}: {error_info['error']}")
        
#         # Update the request store
#         request_store[request_id] = {
#             "request_id": request_id,
#             "status": result.get("status", "completed"),
#             "response": result.get("response", ""),
#             "processing_time": result.get("processing_time", 0),
#             "created_at": request_store[request_id].get("created_at"),
#             "completed_at": asyncio.get_event_loop().time(),
#             "media_files": [f["path"] for f in media_result["saved_files"]] if media_result["saved_files"] else []
#         }
        
#     except Exception as e:
#         logger.error(f"Error processing request {request_id}: {str(e)}", exc_info=True)
#         # Handle errors
#         request_store[request_id] = {
#             "request_id": request_id,
#             "status": "failed",
#             "error": str(e),
#             "created_at": request_store[request_id].get("created_at"),
#             "completed_at": asyncio.get_event_loop().time(),
#         }




# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}


# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=True)