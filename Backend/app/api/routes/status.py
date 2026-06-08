"""
Status Routes — Task status polling and WebSocket real-time updates.
"""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request, Depends, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import asyncio
import json

from app.core.limiter import limiter, STATUS_LIMIT
from app.core.auth import verify_api_key, verify_ws_api_key, validate_task_id
from app.services.task_manager import title_task_manager, TaskStatus

logger = logging.getLogger(__name__)
router = APIRouter()

# WebSocket connection tracking (VULN-06: DoS prevention)
_active_ws_connections = 0
MAX_WS_CONNECTIONS = 100

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
async def get_task_status(
    request: Request, task_id: str,
    _auth: None = Depends(verify_api_key),
):
    """
    Check the progress of a processing task.
    """
    validate_task_id(task_id)
    job = await title_task_manager.get_job_async(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")

    result_data = None
    if job.status in (TaskStatus.COMPLETED, TaskStatus.WAITING_FOR_USER) and job.result:
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
async def websocket_status(websocket: WebSocket, task_id: str, api_key: str = Query(default=None)):
    """
    Real-time status updates via WebSockets + Redis PubSub.
    Authentication via ?api_key=... query parameter.
    """
    # VULN-06: WebSocket authentication
    if not verify_ws_api_key(api_key):
        await websocket.close(code=4001, reason="Unauthorized")
        return
    
    # Validate task_id format
    try:
        validate_task_id(task_id)
    except HTTPException:
        await websocket.close(code=4000, reason="Invalid task ID")
        return
    
    # VULN-06: Connection limit
    global _active_ws_connections
    if _active_ws_connections >= MAX_WS_CONNECTIONS:
        await websocket.close(code=4002, reason="Too many connections")
        return
    
    _active_ws_connections += 1
    await websocket.accept()
    logger.info(f"WebSocket connected for task {task_id} (active: {_active_ws_connections})")
    
    # Try Redis PubSub first
    try:
        import redis.asyncio as aioredis
        from app.core.config import settings
        
        use_redis = False
        async_redis = None
        pubsub = None
        
        if settings.REDIS_URL:
            try:
                async_redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=1)
                await async_redis.ping()
                pubsub = async_redis.pubsub()
                use_redis = True
            except Exception as e:
                logger.warning(f"Redis PubSub unavailable: {e}. Falling back to polling.")
                if async_redis:
                    try:
                        await async_redis.close()
                    except:
                        pass
                async_redis = None
                pubsub = None
                
        if use_redis and pubsub and async_redis:
            channel = f"task:{task_id}"
            await pubsub.subscribe(channel)
            
            try:
                while True:
                    # Non-blocking async check on message channel
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
                    if message and message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            await websocket.send_json(data)
                            
                            # Close on terminal states
                            if data.get('status') in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                                break
                        except Exception as json_e:
                            logger.error(f"Error parsing pubsub message: {json_e}")
                    
                    # Heartbeat & yield event loop
                    await asyncio.sleep(0.1)
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for task {task_id}")
            except Exception as loop_e:
                logger.error(f"Redis subscribe loop error: {loop_e}")
            finally:
                try:
                    await pubsub.unsubscribe(channel)
                    await pubsub.close()
                    await async_redis.close()
                except: pass
        else:
            # Fallback: Internal polling (no Redis) - ASYNC DB POLLING
            logger.info(f"WebSocket fallback to polling for task {task_id}")
            try:
                while True:
                    # Async Poll
                    job = await title_task_manager.get_job_async(task_id)
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
    finally:
        _active_ws_connections -= 1
        logger.info(f"WebSocket closed for task {task_id} (active: {_active_ws_connections})")
