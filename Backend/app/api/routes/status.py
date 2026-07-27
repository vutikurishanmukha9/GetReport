"""
Status Routes — Task status polling and WebSocket real-time updates.
"""
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request, Depends, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import asyncio
import json
import time

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
    Enterprise-grade WebSocket status stream with Redis PubSub / DB polling,
    immediate initial hydration, and 15s ping/pong heartbeats to prevent proxy timeouts.
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
    
    # VULN-06: Connection limit guard
    global _active_ws_connections
    if _active_ws_connections >= MAX_WS_CONNECTIONS:
        await websocket.close(code=4002, reason="Too many connections")
        return
    
    _active_ws_connections += 1
    await websocket.accept()
    logger.info(f"WebSocket connected for task {task_id} (active: {_active_ws_connections})")
    
    # Background task to drain incoming client frames (pings, pongs, close)
    async def _drain_client_frames():
        try:
            while True:
                msg = await websocket.receive_text()
                try:
                    payload = json.loads(msg)
                    if payload.get("type") == "ping":
                        await websocket.send_json({"type": "pong", "timestamp": time.time()})
                except Exception:
                    pass
        except Exception:
            pass  # Socket closed or disconnected

    drain_task = asyncio.create_task(_drain_client_frames())

    try:
        # 1. Immediate Initial Hydration (0ms latency for client)
        job = await title_task_manager.get_job_async(task_id)
        if job:
            initial_payload = {
                "type": "initial_state",
                "task_id": task_id,
                "status": job.status,
                "progress": job.progress or 0,
                "message": job.message or "",
                "result": job.result if job.status in [TaskStatus.COMPLETED, TaskStatus.WAITING_FOR_USER] else None,
                "error": job.error,
            }
            await websocket.send_json(initial_payload)
            if job.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return

        # 2. Try Redis PubSub first
        use_redis = False
        async_redis = None
        pubsub = None

        from app.core.config import settings

        if settings.REDIS_URL:
            try:
                import redis.asyncio as aioredis
                async_redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=1)
                await async_redis.ping()
                pubsub = async_redis.pubsub()
                use_redis = True
            except Exception as e:
                logger.warning(f"Redis PubSub unavailable ({e}). Falling back to async DB polling.")
                if async_redis:
                    try: await async_redis.close()
                    except Exception: pass
                async_redis = None
                pubsub = None

        last_heartbeat = time.time()
        heartbeat_interval = 15.0  # seconds

        if use_redis and pubsub and async_redis:
            channel = f"task:{task_id}"
            await pubsub.subscribe(channel)

            try:
                while True:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.5)
                    if message and message.get('type') == 'message':
                        try:
                            data = json.loads(message['data'])
                            data['type'] = 'status_update'
                            await websocket.send_json(data)
                            last_heartbeat = time.time()
                            
                            if data.get('status') in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                                break
                        except Exception as json_e:
                            logger.error(f"Error parsing pubsub message: {json_e}")

                    # Heartbeat to keep Render/Cloudflare connection alive
                    now = time.time()
                    if now - last_heartbeat >= heartbeat_interval:
                        await websocket.send_json({"type": "ping", "timestamp": now})
                        last_heartbeat = now
                    
                    await asyncio.sleep(0.1)

            except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
                logger.info(f"WebSocket disconnected for task {task_id}")
            except Exception as loop_e:
                logger.error(f"Redis subscribe loop error: {loop_e}")
            finally:
                try:
                    await pubsub.unsubscribe(channel)
                    await pubsub.close()
                    await async_redis.close()
                except Exception: pass
        else:
            # Fallback: Async DB Polling with 15s Heartbeat
            logger.info(f"WebSocket using async DB polling fallback for task {task_id}")
            last_status = None
            last_progress = -1
            last_msg = ""

            try:
                while True:
                    job = await title_task_manager.get_job_async(task_id)
                    now = time.time()

                    if job:
                        cur_status = job.status
                        cur_progress = job.progress or 0
                        cur_msg = job.message or ""

                        # Send update if state changed OR if 15s heartbeat elapsed
                        state_changed = (cur_status != last_status) or (cur_progress != last_progress) or (cur_msg != last_msg)
                        heartbeat_due = (now - last_heartbeat) >= heartbeat_interval

                        if state_changed or heartbeat_due:
                            payload = {
                                "type": "status_update" if state_changed else "ping",
                                "task_id": task_id,
                                "status": cur_status,
                                "progress": cur_progress,
                                "message": cur_msg,
                                "timestamp": now,
                            }
                            if cur_status in [TaskStatus.COMPLETED, TaskStatus.WAITING_FOR_USER] and job.result:
                                payload["result"] = job.result
                            if job.error:
                                payload["error"] = job.error

                            await websocket.send_json(payload)
                            last_status = cur_status
                            last_progress = cur_progress
                            last_msg = cur_msg
                            last_heartbeat = now

                        if cur_status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                            break

                    await asyncio.sleep(1.0)

            except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
                logger.info(f"WebSocket disconnected for task {task_id}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        drain_task.cancel()
        _active_ws_connections -= 1
        logger.info(f"WebSocket closed for task {task_id} (active: {_active_ws_connections})")


# ─── Render Healthcheck Probe ────────────────────────────────────────────────

@router.get("/healthz")
async def health_check():
    """
    Render Uptime Probe & Health Endpoint.
    Monitors process memory, database pool connection, and system status.
    """
    ram_mb = 0.0
    try:
        import psutil
        process = psutil.Process(os.getpid())
        ram_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    except (ImportError, ModuleNotFoundError, Exception):
        ram_mb = 0.0
        
    from app.db import get_async_db_connection
    
    db_healthy = True
    try:
        async with get_async_db_connection() as db:
            await db.execute("SELECT 1")
    except Exception as db_err:
        logger.error(f"Healthcheck DB probe error: {db_err}")
        db_healthy = False

    return {
        "status": "healthy" if db_healthy else "degraded",
        "uptime": "ok",
        "db_connected": db_healthy,
        "process_ram_mb": ram_mb,
        "active_ws_connections": _active_ws_connections
    }


