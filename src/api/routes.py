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
        logger.error(f"Request {request_id} not found. Available request IDs: {list(request_store.keys())}")
        raise HTTPException(status_code=404, detail=f"Request not found: {request_id}")
    
    request_data = request_store[request_id]
    
    # Log the request data for debugging (excluding large media content)
    debug_data = {}
    for key, value in request_data.items():
        if key == "response":
            if isinstance(value, dict):
                debug_data[key] = {k: (f"[{len(v)} chars]" if isinstance(v, str) and len(v) > 100 else v) 
                                  for k, v in value.items()}
            else:
                debug_data[key] = f"[{type(value).__name__}: {len(str(value))} chars]"
        else:
            debug_data[key] = value
    
    logger.info(f"Request data for {request_id}: {debug_data}")
    
    if request_data.get("status") != "completed":
        logger.warning(f"Request {request_id} not completed yet: {request_data.get('status')}")
        raise HTTPException(status_code=400, detail=f"Request not completed yet: {request_data.get('status')}")
    
    response = request_data.get("response", {})
    data_dir = ensure_data_directory()
    
    # Log what's in the media_files list
    if "media_files" in request_data:
        logger.info(f"Media files in request_data: {request_data['media_files']}")
        # Try to use these files if they exist
        for media_file in request_data.get("media_files", []):
            if os.path.exists(media_file):
                file_ext = os.path.splitext(media_file)[1].lower()
                if file_ext in ['.mp3', '.wav', '.ogg']:
                    logger.info(f"Found audio file in media_files: {media_file}")
                    return FileResponse(
                        media_file,
                        media_type=f"audio/{file_ext[1:]}",
                        filename=os.path.basename(media_file)
                    )
                elif file_ext in ['.png', '.jpg', '.jpeg', '.gif']:
                    logger.info(f"Found image file in media_files: {media_file}")
                    return FileResponse(
                        media_file,
                        media_type=f"image/{file_ext[1:]}",
                        filename=os.path.basename(media_file)
                    )
                elif file_ext in ['.mp4', '.webm', '.avi']:
                    logger.info(f"Found video file in media_files: {media_file}")
                    return FileResponse(
                        media_file,
                        media_type=f"video/{file_ext[1:]}",
                        filename=os.path.basename(media_file)
                    )
            else:
                logger.warning(f"Media file does not exist: {media_file}")
    
    # For better debugging, log the response structure
    if isinstance(response, dict):
        logger.info(f"Response keys: {list(response.keys())}")
        
        # Handle different media types in the response dictionary
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
                
                logger.info(f"Found audio_data of length {len(audio_data_str)} chars")
                
                try:
                    decoded_data = base64.b64decode(audio_data)
                    logger.info(f"Successfully decoded audio data, size: {len(decoded_data)} bytes")
                    temp_file.write(decoded_data)
                    temp_file.close()
                except Exception as e:
                    logger.error(f"Error decoding audio data: {str(e)}")
                    raise
                
                # Also save to data directory
                output_path = os.path.join(str(data_dir), f"output_{request_id}.wav")
                try:
                    with open(output_path, 'wb') as f:
                        f.write(decoded_data)
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
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    
    # If we made it here, we might need to check for media files in the data directory
    # Look for files that match the request_id pattern
    possible_media_files = []
    for file_name in os.listdir(data_dir):
        if request_id in file_name:
            file_path = os.path.join(data_dir, file_name)
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in ['.mp3', '.wav', '.ogg', '.png', '.jpg', '.jpeg', '.gif', '.mp4', '.webm', '.avi']:
                possible_media_files.append((file_path, file_ext))
    
    if possible_media_files:
        # Use the first matching file
        file_path, file_ext = possible_media_files[0]
        logger.info(f"Found media file in data directory: {file_path}")
        
        if file_ext in ['.mp3', '.wav', '.ogg']:
            return FileResponse(
                file_path,
                media_type=f"audio/{file_ext[1:]}",
                filename=os.path.basename(file_path)
            )
        elif file_ext in ['.png', '.jpg', '.jpeg', '.gif']:
            return FileResponse(
                file_path,
                media_type=f"image/{file_ext[1:]}",
                filename=os.path.basename(file_path)
            )
        elif file_ext in ['.mp4', '.webm', '.avi']:
            return FileResponse(
                file_path,
                media_type=f"video/{file_ext[1:]}",
                filename=os.path.basename(file_path)
            )
    
    # No media found, provide more detailed error
    if isinstance(response, dict):
        logger.error(f"No media data found in response keys: {list(response.keys())}")
    else:
        logger.error(f"Response is not a dictionary: {type(response)}")
    
    raise HTTPException(
        status_code=404, 
        detail="No media output found for this request. Available response keys: " + 
              (str(list(response.keys())) if isinstance(response, dict) else str(type(response)))
    )





# @router.get("/media/{request_id}")
# async def get_media_output(request_id: str):
#     """
#     Get media output (image, audio, video) from a request.
#     """
#     from ..api.main import request_store
#     from ..utils.media_storage import save_base64_media, ensure_data_directory
#     import tempfile
#     import os
#     import logging
    
#     logger = logging.getLogger("api.media")
    
#     if request_id not in request_store:
#         logger.error(f"Request {request_id} not found. Available request IDs: {list(request_store.keys())}")
#         raise HTTPException(status_code=404, detail=f"Request not found: {request_id}")
    
#     request_data = request_store[request_id]
    
#     # Log the request data for debugging (excluding large media content)
#     debug_data = request_data.copy()
#     if "response" in debug_data and isinstance(debug_data["response"], dict):
#         for key in list(debug_data["response"].keys()):
#             if key.endswith("_data") and isinstance(debug_data["response"][key], str):
#                 debug_data["response"][key] = f"[{len(debug_data['response'][key])} chars]"
#     logger.info(f"Request data for {request_id}: {debug_data}")
    
#     if request_data.get("status") != "completed":
#         logger.warning(f"Request {request_id} not completed yet: {request_data.get('status')}")
#         raise HTTPException(status_code=400, detail=f"Request not completed yet: {request_data.get('status')}")
    
#     response = request_data.get("response", {})
#     data_dir = ensure_data_directory()
    
#     # For better debugging, log the response structure
#     if isinstance(response, dict):
#         logger.info(f"Response keys: {list(response.keys())}")
#     else:
#         logger.info(f"Response type: {type(response)}")
    
#     # Handle different media types
#     if isinstance(response, dict) and "video_data" in response:
#         # Create a temporary file with the video data
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         try:
#             # Extract the actual base64 data if it has a prefix
#             video_data_str = response["video_data"]
#             if ',' in video_data_str:
#                 video_data = video_data_str.split(",")[1]
#             else:
#                 video_data = video_data_str
                
#             temp_file.write(base64.b64decode(video_data))
#             temp_file.close()
            
#             # Also save to data directory
#             output_path = os.path.join(str(data_dir), f"output_{request_id}.mp4")
#             try:
#                 with open(output_path, 'wb') as f:
#                     f.write(base64.b64decode(video_data))
#                 logger.info(f"Saved video to data directory: {output_path}")
#             except Exception as e:
#                 logger.error(f"Failed to save video to data directory: {str(e)}")
            
#             return FileResponse(
#                 temp_file.name,
#                 media_type="video/mp4",
#                 filename=f"output_{request_id}.mp4"
#             )
#         except Exception as e:
#             logger.error(f"Error processing video: {str(e)}")
#             raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
#     elif isinstance(response, dict) and "image_data" in response:
#         # Create a temporary file with the image data
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
#         try:
#             # Extract the actual base64 data if it has a prefix
#             image_data_str = response["image_data"]
#             if ',' in image_data_str:
#                 image_data = image_data_str.split(",")[1]
#             else:
#                 image_data = image_data_str
                
#             temp_file.write(base64.b64decode(image_data))
#             temp_file.close()
            
#             # Also save to data directory
#             output_path = os.path.join(str(data_dir), f"output_{request_id}.png")
#             try:
#                 with open(output_path, 'wb') as f:
#                     f.write(base64.b64decode(image_data))
#                 logger.info(f"Saved image to data directory: {output_path}")
#             except Exception as e:
#                 logger.error(f"Failed to save image to data directory: {str(e)}")
            
#             return FileResponse(
#                 temp_file.name,
#                 media_type="image/png",
#                 filename=f"output_{request_id}.png"
#             )
#         except Exception as e:
#             logger.error(f"Error processing image: {str(e)}")
#             raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
#     elif isinstance(response, dict) and "audio_data" in response:
#         # Create a temporary file with the audio data
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#         try:
#             # Extract the actual base64 data if it has a prefix
#             audio_data_str = response["audio_data"]
#             if ',' in audio_data_str:
#                 audio_data = audio_data_str.split(",")[1]
#             else:
#                 audio_data = audio_data_str
                
#             temp_file.write(base64.b64decode(audio_data))
#             temp_file.close()
            
#             # Also save to data directory
#             output_path = os.path.join(str(data_dir), f"output_{request_id}.wav")
#             try:
#                 with open(output_path, 'wb') as f:
#                     f.write(base64.b64decode(audio_data))
#                 logger.info(f"Saved audio to data directory: {output_path}")
#             except Exception as e:
#                 logger.error(f"Failed to save audio to data directory: {str(e)}")
            
#             return FileResponse(
#                 temp_file.name,
#                 media_type="audio/wav",
#                 filename=f"output_{request_id}.wav"
#             )
#         except Exception as e:
#             logger.error(f"Error processing audio: {str(e)}")
#             raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    
#     else:
#         # No media found, provide more detailed error
#         if isinstance(response, dict):
#             logger.error(f"No media data found in response keys: {list(response.keys())}")
#         else:
#             logger.error(f"Response is not a dictionary: {type(response)}")
        
#         raise HTTPException(
#             status_code=404, 
#             detail="No media output found for this request. Available response keys: " + 
#                   (str(list(response.keys())) if isinstance(response, dict) else "none")
#         )





# @router.get("/media/{request_id}")
# async def get_media_output(request_id: str):
#     """
#     Get media output (image, audio, video) from a request.
#     """
#     from ..api.main import request_store
#     from ..utils.media_storage import save_base64_media, ensure_data_directory
#     import tempfile
#     import os
#     import logging
    
#     logger = logging.getLogger("api.media")
    
#     if request_id not in request_store:
#         raise HTTPException(status_code=404, detail="Request not found")
    
#     request_data = request_store[request_id]
#     if request_data.get("status") != "completed":
#         raise HTTPException(status_code=400, detail="Request not completed yet")
    
#     response = request_data.get("response", {})
#     data_dir = ensure_data_directory()
    
#     # Handle different media types
#     if "video_data" in response:
#         # Create a temporary file with the video data
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         try:
#             # Extract the actual base64 data if it has a prefix
#             video_data_str = response["video_data"]
#             if ',' in video_data_str:
#                 video_data = video_data_str.split(",")[1]
#             else:
#                 video_data = video_data_str
                
#             temp_file.write(base64.b64decode(video_data))
#             temp_file.close()
            
#             # Also save to data directory
#             output_path = os.path.join(str(data_dir), f"output_{request_id}.mp4")
#             try:
#                 with open(output_path, 'wb') as f:
#                     f.write(base64.b64decode(video_data))
#                 logger.info(f"Saved video to data directory: {output_path}")
#             except Exception as e:
#                 logger.error(f"Failed to save video to data directory: {str(e)}")
            
#             return FileResponse(
#                 temp_file.name,
#                 media_type="video/mp4",
#                 filename=f"output_{request_id}.mp4"
#             )
#         except Exception as e:
#             logger.error(f"Error processing video: {str(e)}")
#             raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
#     elif "image_data" in response:
#         # Create a temporary file with the image data
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
#         try:
#             # Extract the actual base64 data if it has a prefix
#             image_data_str = response["image_data"]
#             if ',' in image_data_str:
#                 image_data = image_data_str.split(",")[1]
#             else:
#                 image_data = image_data_str
                
#             temp_file.write(base64.b64decode(image_data))
#             temp_file.close()
            
#             # Also save to data directory
#             output_path = os.path.join(str(data_dir), f"output_{request_id}.png")
#             try:
#                 with open(output_path, 'wb') as f:
#                     f.write(base64.b64decode(image_data))
#                 logger.info(f"Saved image to data directory: {output_path}")
#             except Exception as e:
#                 logger.error(f"Failed to save image to data directory: {str(e)}")
            
#             return FileResponse(
#                 temp_file.name,
#                 media_type="image/png",
#                 filename=f"output_{request_id}.png"
#             )
#         except Exception as e:
#             logger.error(f"Error processing image: {str(e)}")
#             raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
#     elif "audio_data" in response:
#         # Create a temporary file with the audio data
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#         try:
#             # Extract the actual base64 data if it has a prefix
#             audio_data_str = response["audio_data"]
#             if ',' in audio_data_str:
#                 audio_data = audio_data_str.split(",")[1]
#             else:
#                 audio_data = audio_data_str
                
#             temp_file.write(base64.b64decode(audio_data))
#             temp_file.close()
            
#             # Also save to data directory
#             output_path = os.path.join(str(data_dir), f"output_{request_id}.wav")
#             try:
#                 with open(output_path, 'wb') as f:
#                     f.write(base64.b64decode(audio_data))
#                 logger.info(f"Saved audio to data directory: {output_path}")
#             except Exception as e:
#                 logger.error(f"Failed to save audio to data directory: {str(e)}")
            
#             return FileResponse(
#                 temp_file.name,
#                 media_type="audio/wav",
#                 filename=f"output_{request_id}.wav"
#             )
#         except Exception as e:
#             logger.error(f"Error processing audio: {str(e)}")
#             raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    
#     else:
#         # No media found
#         raise HTTPException(status_code=404, detail="No media output found for this request")