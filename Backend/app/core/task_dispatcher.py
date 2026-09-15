"""
Unified Lean Task Dispatcher for GetReport.
Implements dual-mode background task execution:
  1. ARQ Mode (Production with Redis): Dispatches async tasks to an ARQ worker pool (~25MB RAM).
  2. In-Memory Mode (Local Dev without Redis): Dispatches coroutines directly into asyncio event loop
     with zero external software, zero socket timeouts, and zero overhead.
"""
import asyncio
import logging
import threading
from typing import Any, Callable, Dict, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# Registry of task functions: name -> callable
_TASK_REGISTRY: Dict[str, Callable] = {}
_arq_pool: Optional[Any] = None
_redis_available: Optional[bool] = None
_pool_lock = asyncio.Lock() if hasattr(asyncio, "Lock") else None


def register_task(name: str):
    """Decorator to register a task callable by string name."""
    def decorator(func: Callable):
        _TASK_REGISTRY[name] = func
        return func
    return decorator


def check_redis_available() -> bool:
    """
    Check if Redis is reachable with a fast 0.5s timeout.
    Result is cached to prevent repeated connection attempts in offline environments.
    """
    global _redis_available
    if _redis_available is not None:
        return _redis_available

    try:
        import redis
        client = redis.from_url(settings.REDIS_URL, socket_connect_timeout=0.2, socket_timeout=0.2)
        client.ping()
        _redis_available = True
        logger.info("Task Dispatcher: Redis connection verified at %s (ARQ Mode active).", settings.REDIS_URL)
    except Exception:
        _redis_available = False
        logger.info("Task Dispatcher: Redis not reachable. Falling back to in-memory native asyncio task runner.")

    return _redis_available


async def _get_arq_pool():
    """Lazily initialize ARQ connection pool when Redis is available."""
    global _arq_pool
    if _arq_pool is not None:
        return _arq_pool

    try:
        from arq import create_pool
        from arq.connections import RedisSettings
        _arq_pool = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))
        logger.info("Task Dispatcher: ARQ worker connection pool created.")
        return _arq_pool
    except Exception as e:
        logger.warning("Task Dispatcher: Failed to create ARQ pool (%s). Using in-memory runner.", e)
        return None


def _ensure_registry_loaded():
    """Ensure all default tasks are loaded into _TASK_REGISTRY."""
    if not _TASK_REGISTRY:
        try:
            import app.tasks_arq  # noqa: F401
        except ImportError as e:
            logger.debug("Could not import app.tasks_arq: %s", e)


def _call_handler(handler: Callable, *args: Any, **kwargs: Any):
    """Inspect handler signature to pass ctx if expected by ARQ task definitions."""
    import inspect
    sig = inspect.signature(handler)
    params = list(sig.parameters.keys())
    if params and params[0] in ("ctx", "context"):
        ctx = {"job_id": "in-memory", "redis": None}
        return handler(ctx, *args, **kwargs)
    return handler(*args, **kwargs)


async def dispatch_task(task_name: str, *args: Any, **kwargs: Any) -> str:
    """
    Dispatch a background task asynchronously.
    Routes to ARQ Redis worker if Redis is reachable; otherwise executes
    concurrently in the local event loop via asyncio.create_task.
    """
    redis_online = check_redis_available()

    if redis_online:
        pool = await _get_arq_pool()
        if pool:
            try:
                job = await pool.enqueue_job(task_name, *args, **kwargs)
                job_id = job.job_id if job else "queued"
                logger.info("Dispatched task '%s' to ARQ worker (job_id=%s)", task_name, job_id)
                return job_id
            except Exception as arq_err:
                logger.warning("ARQ enqueue failed (%s). Falling back to in-memory execution.", arq_err)

    # In-memory execution fallback (Local Dev & zero-Redis environments)
    _ensure_registry_loaded()
    handler = _TASK_REGISTRY.get(task_name)
    if not handler:
        raise ValueError(f"Task '{task_name}' is not registered in the task dispatcher.")

    async def _runner():
        try:
            import inspect
            sig = inspect.signature(handler)
            params = list(sig.parameters.keys())
            ctx = {"job_id": "in-memory", "redis": None} if params and params[0] in ("ctx", "context") else None

            if asyncio.iscoroutinefunction(handler):
                if ctx is not None:
                    await handler(ctx, *args, **kwargs)
                else:
                    await handler(*args, **kwargs)
            else:
                if ctx is not None:
                    await asyncio.to_thread(handler, ctx, *args, **kwargs)
                else:
                    await asyncio.to_thread(handler, *args, **kwargs)
        except Exception as err:
            logger.error("In-memory task '%s' failed: %s", task_name, err, exc_info=True)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_runner())
        logger.info("Dispatched task '%s' to native in-memory asyncio task runner.", task_name)
        return "in-memory"
    except RuntimeError:
        # If called from a sync context where no loop is running
        return dispatch_task_sync(task_name, *args, **kwargs)


def dispatch_task_sync(task_name: str, *args: Any, **kwargs: Any) -> str:
    """
    Synchronous entry point for dispatching tasks from sync route handlers or legacy code.
    Safely schedules into the running event loop or a dedicated background thread.
    """
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            loop.create_task(dispatch_task(task_name, *args, **kwargs))
            return "in-memory"
    except RuntimeError:
        pass

    # No running event loop in current thread: dispatch via worker thread
    _ensure_registry_loaded()
    handler = _TASK_REGISTRY.get(task_name)
    if not handler:
        raise ValueError(f"Task '{task_name}' is not registered in the task dispatcher.")

    def _thread_worker():
        try:
            import inspect
            sig = inspect.signature(handler)
            params = list(sig.parameters.keys())
            ctx = {"job_id": "in-memory", "redis": None} if params and params[0] in ("ctx", "context") else None

            if asyncio.iscoroutinefunction(handler):
                if ctx is not None:
                    asyncio.run(handler(ctx, *args, **kwargs))
                else:
                    asyncio.run(handler(*args, **kwargs))
            else:
                if ctx is not None:
                    handler(ctx, *args, **kwargs)
                else:
                    handler(*args, **kwargs)
        except Exception as err:
            logger.error("Background thread task '%s' failed: %s", task_name, err, exc_info=True)

    thread = threading.Thread(target=_thread_worker, daemon=True)
    thread.start()
    logger.info("Dispatched task '%s' via daemon thread.", task_name)
    return "threaded"

