import sys
import os
import json
import uuid

# Add Backend to path
sys.path.append(os.path.join(os.getcwd(), "Backend"))

# Mocking settings to ensure we use Local Storage for test
from app.core.config import settings
settings.STORAGE_TYPE = "local"

from app.services.task_manager import title_task_manager, TaskStatus
from app.services.storage import get_storage_provider

def test_optimization():
    print("Testing Database Optimization (Storage Provider Integration)...")
    
    # 1. Create Job
    task_id = title_task_manager.create_job("test_data.csv")
    print(f"✓ Job Created: {task_id}")
    
    # 2. Simulate Result Generation
    dummy_result = {
        "summary": {"mean": 100, "std": 15},
        "columns": ["A", "B", "C"],
        "large_blob": "x" * 1000  # Simulate heavy blob
    }
    
    # 3. Complete Job (Should save to storage)
    title_task_manager.complete_job(task_id, dummy_result)
    print("✓ Job Completed")
    
    # 4. Verify DB State (via TaskManager)
    # This checks if retrieval works (loading from file/storage)
    job = title_task_manager.get_job(task_id)
    
    if not job:
        print("✗ Failed to retrieve job")
        sys.exit(1)
        
    print(f"✓ Retrieved Job Status: {job.status}")
    print(f"✓ Retrieved Result Path: {job.result_path}")
    
    # 5. Verify Content
    if job.result and job.result["summary"]["mean"] == 100:
        print("✓ Result content matches (loaded from storage correctly)")
    else:
        print("✗ Result content mismatch or missing")
        sys.exit(1)
        
    # 6. Verify File Exists in Storage
    storage = get_storage_provider()
    local_path = storage.get_absolute_path(job.result_path)
    if os.path.exists(local_path):
        print(f"✓ Storage file exists at: {local_path}")
    else:
        print(f"✗ Storage file missing at: {local_path}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_optimization()
    except Exception as e:
        print(f"✗ Test Failed: {e}")
        import traceback
        traceback.print_exc()
