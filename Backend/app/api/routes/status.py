"""
Status Routes — Task status polling and WebSocket real-time updates.
"""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import asyncio
import json

from app.core.limiter import limiter, STATUS_LIMIT
from app.services.task_manager import title_task_manager, TaskStatus

logger = logging.getLogger(__name__)
router = APIRouter()

class StatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    report_download_url: Optional[str] = None

@router.get("/status/{task_id}", response_model=StatusResponse)
@limiter.limit(STATUS_LIMIT)
async def get_task_status(request: Request, task_id: str):
    """
    Check the progress of a processing task.
    """
    job = title_task_manager.get_job(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")

    result_data = None
    if job.status == TaskStatus.COMPLETED and job.result:
        result_data = job.result

    return StatusResponse(
        task_id=task_id,
        status=job.status,
        progress=job.progress or 0,
        message=job.message or "",
        result=result_data,
        error=job.error,
        report_download_url=f"/api/jobs/{task_id}/report" if job.report_path else None
    )


@router.websocket("/ws/status/{task_id}")
async def websocket_status(websocket: WebSocket, task_id: str):
    """
    Real-time status updates via WebSockets + Redis PubSub.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for task {task_id}")
    
    # Try Redis PubSub first
    try:
        from app.services.task_manager import redis_client
        if redis_client:
            pubsub = redis_client.pubsub()
            channel = f"task:{task_id}"
            pubsub.subscribe(channel)
            
            try:
                while True:
                    message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if message and message['type'] == 'message':
                        data = json.loads(message['data'])
                        await websocket.send_json(data)
                        
                        # Close on terminal states
                        if data.get('status') in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                            break
                    
                    # Heartbeat
                    await asyncio.sleep(0.5)
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for task {task_id}")
            finally:
                pubsub.unsubscribe(channel)
        else:
            # Fallback: Internal polling (no Redis)
            logger.info(f"WebSocket fallback to polling for task {task_id}")
            try:
                while True:
                    job = title_task_manager.get_job(task_id)
                    if job:
                        await websocket.send_json({
                            "task_id": task_id,
                            "status": job.status,
                            "progress": job.progress or 0,
                            "message": job.message or "",
                        })
                        
                        if job.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                            break
                    
                    await asyncio.sleep(2)
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for task {task_id}")
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass
