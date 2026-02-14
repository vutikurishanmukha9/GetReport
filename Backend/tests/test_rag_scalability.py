import sys
import os
import asyncio
from unittest.mock import MagicMock, patch

# Add Backend to path
sys.path.append(os.path.join(os.getcwd(), "Backend"))

from app.core.config import settings
from app.services.rag_service import EnhancedRAGService

# Mock Config to simulate Prod vs Local
def test_rag_factory_logic():
    print("Testing RAG Store Selection Logic...")
    
    # Pre-requisite: Set API Key to enable service
    settings.OPENAI_API_KEY = "sk-mock-key"
    
    # 1. Test Local (No DATABASE_URL)
    settings.DATABASE_URL = None
    rag = EnhancedRAGService()
    
    # Mock dependencies
    rag._get_embeddings = AsyncOpenAI_mock = MagicMock()
    AsyncOpenAI_mock.return_value = [[0.1, 0.2]] # async? No, _get_embeddings is async
    rag.cache.set = MagicMock()
    
    async def mock_embed(texts):
        return [[0.1] * 1536] * len(texts)
    rag._get_embeddings = mock_embed
    
    # Run Ingest
    print("-> Testing Local Mode (In-Memory)...")
    asyncio.run(rag.ingest_report("task_local", "some text"))
    
    # Verify Cache Set was called (implies SimpleVectorStore usage)
    if rag.cache.set.called:
        print("✓ Local Mode used In-Memory Cache (SimpleVectorStore)")
    else:
        print("✗ Local Mode failed to use In-Memory Cache")

    # 2. Test Prod (With DATABASE_URL)
    settings.DATABASE_URL = "postgresql://user:pass@localhost/db"
    
    # We need to mock PostgresVectorStore to avoid actual DB connection attempt
    with patch("app.services.rag_service.PostgresVectorStore") as MockStore:
        print("-> Testing Prod Mode (Postgres)...")
        rag = EnhancedRAGService()
        rag._get_embeddings = mock_embed
        
        # Reset cache mock to ensure it's NOT called
        rag.cache.set = MagicMock()
        
        asyncio.run(rag.ingest_report("task_prod", "some text"))
        
        if MockStore.called:
            print("✓ Prod Mode instantiated PostgresVectorStore")
            if MockStore.return_value.add_texts.called:
                 print("✓ Prod Mode called add_texts on PostgresVectorStore")
            else:
                 print("✗ Prod Mode did not call add_texts")
        else:
            print("✗ Prod Mode did not use PostgresVectorStore")
            
        if not rag.cache.set.called:
             print("✓ Prod Mode bypassed In-Memory Cache (Correct)")
        else:
             print("✗ Prod Mode incorrectly used In-Memory Cache")

if __name__ == "__main__":
    try:
        test_rag_factory_logic()
    except Exception as e:
        print(f"✗ Test Failed: {e}")
        import traceback
        traceback.print_exc()
