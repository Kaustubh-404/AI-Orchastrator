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
    from ..utils.media_storage import save_media_response, ensure_data_directory
    
    logger = logging.getLogger("api.request_processor")
    
    # Make sure data directory exists
    data_dir = ensure_data_directory()
    logger.info(f"Using data directory: {data_dir}")
    
    try:
        # Process the request with the orchestrator
        logger.info(f"Processing request {request_id}: {content[:100]}...")
        result = await orchestrator.process_request(content, context)
        
        # Save any media in the response to the data directory
        media_result = save_media_response(result, f"request_{request_id}")
        
        # Log the results of media saving
        if media_result["saved_files"]:
            for file_info in media_result["saved_files"]:
                logger.info(f"Saved {file_info['type']} to {file_info['path']}")
        
        if media_result["errors"]:
            for error_info in media_result["errors"]:
                logger.error(f"Failed to save {error_info['type']}: {error_info['error']}")
        
        # Update the request store
        request_store[request_id] = {
            "request_id": request_id,
            "status": result.get("status", "completed"),
            "response": result.get("response", ""),
            "processing_time": result.get("processing_time", 0),
            "created_at": request_store[request_id].get("created_at"),
            "completed_at": asyncio.get_event_loop().time(),
            "media_files": [f["path"] for f in media_result["saved_files"]] if media_result["saved_files"] else []
        }
        
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