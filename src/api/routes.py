from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status, File, UploadFile, Form
from typing import Dict, Any, List
from fastapi.responses import FileResponse
import os
import tempfile
import base64
import json
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

@router.post("/upload", response_model=Dict[str, str])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file (image, audio, etc.) to be used in requests.
    """
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # For images and audio, convert to base64 for easier handling
    content_type = file.content_type
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    
    base64_str = base64.b64encode(file_bytes).decode("utf-8")
    data_uri = f"data:{content_type};base64,{base64_str}"
    
    return {
        "file_id": os.path.basename(temp_dir),
        "filename": file.filename,
        "content_type": content_type,
        "data_uri": data_uri
    }

@router.post("/requests-with-media", response_model=RequestStatus)
async def create_request_with_media(
    content: str = Form(...),
    files: List[UploadFile] = File(None),
    background_tasks: BackgroundTasks = None,
    orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """
    Submit a new request with media files to the orchestrator.
    """
    from ..api.main import process_request_background, request_store
    import uuid
    import asyncio
    
    request_id = str(uuid.uuid4())
    
    # Process uploaded files
    media_data = {}
    if files:
        for file in files:
            file_bytes = await file.read()
            content_type = file.content_type
            base64_str = base64.b64encode(file_bytes).decode("utf-8")
            data_uri = f"data:{content_type};base64,{base64_str}"
            
            if content_type.startswith("image/"):
                if "images" not in media_data:
                    media_data["images"] = []
                media_data["images"].append(data_uri)
            elif content_type.startswith("audio/"):
                media_data["audio"] = data_uri
            elif content_type.startswith("video/"):
                media_data["video"] = data_uri
    
    # Create context with media data
    context = {"media_data": media_data}
    
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
        content=content,
        context=context,
    )
    
    return RequestStatus(
        request_id=request_id,
        status="processing",
        message="Request with media accepted for processing",
    )

@router.get("/media/{request_id}")
async def get_media_output(request_id: str):
    """
    Get media output (image, audio, video) from a request.
    """
    from ..api.main import request_store
    
    if request_id not in request_store:
        raise HTTPException(status_code=404, detail="Request not found")
    
    request_data = request_store[request_id]
    if request_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Request not completed yet")
    
    response = request_data.get("response", {})
    
    # Handle different media types
    if "video_data" in response:
        # Create a temporary file with the video data
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_data = response["video_data"].split(",")[1]
        temp_file.write(base64.b64decode(video_data))
        temp_file.close()
        
        return FileResponse(
            temp_file.name,
            media_type="video/mp4",
            filename=f"output_{request_id}.mp4"
        )
    
    elif "image_data" in response:
        # Create a temporary file with the image data
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image_data = response["image_data"].split(",")[1]
        temp_file.write(base64.b64decode(image_data))
        temp_file.close()
        
        return FileResponse(
            temp_file.name,
            media_type="image/png",
            filename=f"output_{request_id}.png"
        )
    
    elif "audio_data" in response:
        # Create a temporary file with the audio data
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_data = response["audio_data"].split(",")[1]
        temp_file.write(base64.b64decode(audio_data))
        temp_file.close()
        
        return FileResponse(
            temp_file.name,
            media_type="audio/wav",
            filename=f"output_{request_id}.wav"
        )
    
    else:
        # No media found
        raise HTTPException(status_code=404, detail="No media output found for this request")