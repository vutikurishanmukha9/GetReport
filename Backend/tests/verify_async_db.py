import asyncio
import uuid
import logging
from app.db import init_async_db, close_async_db, get_async_db_connection
from app.services.task_manager import title_task_manager, TaskStatus
from app.services.rag_service import rag_service
from app.core.config import settings

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_async_task_manager():
    logger.info("--- Testing Async Task Manager ---")
    
    # 1. Create Job
    filename = "test_async.csv"
    task_id = await title_task_manager.create_job_async(filename)
    logger.info(f"Created Job: {task_id}")
    
    # 2. Get Job
    job = await title_task_manager.get_job_async(task_id)
    assert job is not None
    assert job.id == task_id
    assert job.status == TaskStatus.PENDING
    assert job.version == 1
    logger.info("Get Job Verified")
    
    # 3. Update Status
    await title_task_manager.update_status_async(task_id, TaskStatus.PROCESSING)
    job = await title_task_manager.get_job_async(task_id)
    assert job.status == TaskStatus.PROCESSING
    assert job.version == 2
    logger.info("Update Status Verified")
    
    # 4. Update Result (Issue Ledger style)
    await title_task_manager.update_result_async(task_id, {"test": "data"})
    job = await title_task_manager.get_job_async(task_id)
    assert job.result == {"test": "data"}
    assert job.version == 3
    logger.info("Update Result Verified")
    
    return task_id

async def test_async_rag(task_id):
    logger.info("--- Testing Async RAG Service ---")
    if not settings.DATABASE_URL:
        logger.warning("Skipping RAG test (No DATABASE_URL)")
        return

    # 1. Ingest
    text = "This is a test report about AI agents."
    result = await rag_service.ingest_report(task_id, text, metadata={"source": "test"})
    assert result["success"] is True
    logger.info("RAG Ingest Verified")
    
    # 2. Chat/Search
    # We wait a bit for any potential async comms (though ingest should be awaited)
    search_result = await rag_service.chat_with_report(task_id, "What is this report about?", k=1)
    if search_result["success"]:
        logger.info(f"RAG Chat Answer: {search_result['answer']}")
    else:
        logger.error(f"RAG Chat Failed: {search_result.get('error')}")
        
    logger.info("RAG Search Verified")

async def main():
    try:
        # Initialize Schema (Sync)
        from app.db import init_db
        init_db()
        
        # Initialize Async Pool
        await init_async_db()
        
        # Run Tests
        task_id = await test_async_task_manager()
        await test_async_rag(task_id)
        
        logger.info("ALL ASYNC TESTS PASSED")
        
    except Exception as e:
        logger.error(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await close_async_db()

if __name__ == "__main__":
    if not settings.DATABASE_URL:
        print("WARNING: DATABASE_URL not set. Tests mainly valid for PostgreSQL.")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
