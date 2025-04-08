from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List

from .models import RequestInput, RequestStatus, AgentInfo
from ..core.orchestrator import Orchestrator

# Create the router
router = APIRouter(prefix="/api/v1")


async def get_orchestrator():
    """Dependency to get the orchestrator instance."""
    # In a real application, this would likely be a singleton or retrieved from a context
    from ..api.main import orchestrator
    return orchestrator


@router.post("/requests", response_model=RequestStatus)
async def create_request(
    request_input: RequestInput, 
    background_tasks: BackgroundTasks,
    orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """
    Submit a new request to the orchestrator.
    """
    from ..api.main import process_request_background, request_store
    import uuid
    import asyncio
    
    request_id = str(uuid.uuid4())
    
    # Store initial status
    request_store[request_id] = {
        "request_id": request_id,
        "status": "processing",
        "created_at": asyncio.get_event_loop().time(),
    }
    
    # Process the request in the background
    background_tasks.add_task(
        process_request_background,
        request_id=request_id,
        content=request_input.content,
        context=request_input.context,
    )
    
    return RequestStatus(
        request_id=request_id,
        status="processing",
        message="Request accepted for processing",
    )


@router.get("/requests/{request_id}", response_model=RequestStatus)
async def get_request_status(request_id: str):
    """
    Get the status of a request.
    """
    from ..api.main import request_store
    
    if request_id not in request_store:
        raise HTTPException(status_code=404, detail="Request not found")
    
    request_data = request_store[request_id]
    
    return RequestStatus(
        request_id=request_id,
        status=request_data.get("status", "unknown"),
        message=f"Request {request_data.get('status', 'unknown')}",
        response=request_data.get("response"),
        error=request_data.get("error"),
        processing_time=request_data.get("processing_time"),
    )


@router.get("/requests", response_model=List[RequestStatus])
async def list_requests():
    """
    List all requests.
    """
    from ..api.main import request_store
    
    return [
        RequestStatus(
            request_id=req_id,
            status=req_data.get("status", "unknown"),
            message=f"Request {req_data.get('status', 'unknown')}",
            created_at=req_data.get("created_at"),
            completed_at=req_data.get("completed_at"),
            processing_time=req_data.get("processing_time"),
        )
        for req_id, req_data in request_store.items()
    ]


@router.get("/agents", response_model=List[AgentInfo])
async def list_agents(orchestrator: Orchestrator = Depends(get_orchestrator)):
    """
    List all registered agents.
    """
    agents = orchestrator.agent_registry.get_all_agents()
    
    return [
        AgentInfo(
            agent_id=agent["agent_id"],
            name=agent["name"],
            description=agent["description"],
            capabilities=agent["capabilities"],
            metrics=agent["metrics"],
        )
        for agent in agents
    ]