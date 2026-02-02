
import sys
import os
import asyncio
import logging

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ManualCheck")

async def test_rag_service():
    print("--- STARTING RAG SERVICE CHECK ---")
    
    try:
        print("1. Attempting Import...")
        from app.services.rag_service import rag_service, EnhancedRAGService
        print("   [PASS] Import successful.")
        
        print(f"2. Checking Status: Enabled={rag_service.enabled}")
        
        print("3. Testing Ingestion (Dry Run)...")
        # Should return success=False or skip if disabled, but NOT crash
        res_ingest = await rag_service.ingest_report("test_task_id", "Some dummy text content")
        print(f"   Result: {res_ingest}")
        
        print("4. Testing Chat (Dry Run)...")
        res_chat = await rag_service.chat_with_report("test_task_id", "Hello?")
        print(f"   Result: {res_chat}")
        
        print("--- RAG SERVICE CHECK PASSED ---")
        return True
        
    except ImportError as e:
        print(f"!!! FATAL: Import Failed: {e}")
        return False
    except Exception as e:
        print(f"!!! FATAL: Runtime Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_rag_service())
