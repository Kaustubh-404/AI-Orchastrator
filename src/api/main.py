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


# Initialize agents on startup
@app.on_event("startup")
async def startup_event():
    # Register default agents
    text_agent = TextGenerationAgent()
    reasoning_agent = ReasoningAgent()
    orchestrator.register_agent(text_agent)
    orchestrator.register_agent(reasoning_agent)
    
    # Start health checks
    agent_registry.start_health_checks()


# Include API router
app.include_router(api_router)


async def process_request_background(request_id: str, content: str, context: Optional[Dict[str, Any]] = None):
    """
    Process a request in the background.
    """
    try:
        # Process the request with the orchestrator
        result = await orchestrator.process_request(content, context)
        
        # Update the request store
        request_store[request_id] = {
            "request_id": request_id,
            "status": result.get("status", "completed"),
            "response": result.get("response", ""),
            "processing_time": result.get("processing_time", 0),
            "created_at": request_store[request_id].get("created_at"),
            "completed_at": asyncio.get_event_loop().time(),
        }
        
    except Exception as e:
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