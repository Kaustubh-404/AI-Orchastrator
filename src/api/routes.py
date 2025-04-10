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


# @router.get("/requests/{request_id}", response_model=RequestStatus)
# async def get_request_status(request_id: str):
#     """
#     Get the status of a request.
#     """
#     from ..api.main import request_store
    
#     if request_id not in request_store:
#         raise HTTPException(status_code=404, detail="Request not found")
    
#     request_data = request_store[request_id]
    
#     return RequestStatus(
#         request_id=request_id,
#         status=request_data.get("status", "unknown"),
#         message=f"Request {request_data.get('status', 'unknown')}",
#         response=request_data.get("response"),
#         error=request_data.get("error"),
#         processing_time=request_data.get("processing_time"),
#     )


@router.get("/requests/{request_id}", response_model=RequestStatus)
async def get_request_status(request_id: str):
    """
    Get the status of a request.
    """
    from ..api.main import request_store
    
    if request_id not in request_store:
        raise HTTPException(status_code=404, detail="Request not found")
    
    request_data = request_store[request_id]
    
    # For failed requests, include more detailed error info
    if request_data.get("status") == "failed":
        error_detail = request_data.get("error", "Unknown error")
        
        # Check if there are more detailed errors in the execution results
        if "stages" in request_data and "execution" in request_data["stages"]:
            execution = request_data["stages"]["execution"]
            if "result" in execution and "task_results" in execution["result"]:
                task_errors = []
                for task_id, task_result in execution["result"]["task_results"].items():
                    if task_result.get("status") == "failed":
                        task_errors.append({
                            "task_id": task_id,
                            "error": task_result.get("error", "Unknown task error")
                        })
                if task_errors:
                    error_detail = f"{error_detail}. Task errors: {task_errors}"
    
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

# This is a modified version of the get_media_output function from src/api/routes.py
# to be integrated into your existing codebase

@router.get("/media/{request_id}")
async def get_media_output(request_id: str):
    """
    Get media output (image, audio, video) from a request.
    """
    from ..api.main import request_store
    from ..utils.media_storage import save_base64_media, ensure_data_directory
    import tempfile
    import os
    import logging
    
    logger = logging.getLogger("api.media")
    
    if request_id not in request_store:
        raise HTTPException(status_code=404, detail="Request not found")
    
    request_data = request_store[request_id]
    if request_data.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Request not completed yet")
    
    response = request_data.get("response", {})
    data_dir = ensure_data_directory()
    
    # Handle different media types
    if "video_data" in response:
        # Create a temporary file with the video data
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        try:
            # Extract the actual base64 data if it has a prefix
            video_data_str = response["video_data"]
            if ',' in video_data_str:
                video_data = video_data_str.split(",")[1]
            else:
                video_data = video_data_str
                
            temp_file.write(base64.b64decode(video_data))
            temp_file.close()
            
            # Also save to data directory
            output_path = os.path.join(str(data_dir), f"output_{request_id}.mp4")
            try:
                with open(output_path, 'wb') as f:
                    f.write(base64.b64decode(video_data))
                logger.info(f"Saved video to data directory: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save video to data directory: {str(e)}")
            
            return FileResponse(
                temp_file.name,
                media_type="video/mp4",
                filename=f"output_{request_id}.mp4"
            )
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    elif "image_data" in response:
        # Create a temporary file with the image data
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        try:
            # Extract the actual base64 data if it has a prefix
            image_data_str = response["image_data"]
            if ',' in image_data_str:
                image_data = image_data_str.split(",")[1]
            else:
                image_data = image_data_str
                
            temp_file.write(base64.b64decode(image_data))
            temp_file.close()
            
            # Also save to data directory
            output_path = os.path.join(str(data_dir), f"output_{request_id}.png")
            try:
                with open(output_path, 'wb') as f:
                    f.write(base64.b64decode(image_data))
                logger.info(f"Saved image to data directory: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save image to data directory: {str(e)}")
            
            return FileResponse(
                temp_file.name,
                media_type="image/png",
                filename=f"output_{request_id}.png"
            )
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    elif "audio_data" in response:
        # Create a temporary file with the audio data
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            # Extract the actual base64 data if it has a prefix
            audio_data_str = response["audio_data"]
            if ',' in audio_data_str:
                audio_data = audio_data_str.split(",")[1]
            else:
                audio_data = audio_data_str
                
            temp_file.write(base64.b64decode(audio_data))
            temp_file.close()
            
            # Also save to data directory
            output_path = os.path.join(str(data_dir), f"output_{request_id}.wav")
            try:
                with open(output_path, 'wb') as f:
                    f.write(base64.b64decode(audio_data))
                logger.info(f"Saved audio to data directory: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save audio to data directory: {str(e)}")
            
            return FileResponse(
                temp_file.name,
                media_type="audio/wav",
                filename=f"output_{request_id}.wav"
            )
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    
    else:
        # No media found
        raise HTTPException(status_code=404, detail="No media output found for this request")