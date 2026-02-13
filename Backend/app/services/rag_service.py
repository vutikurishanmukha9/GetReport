import logging
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

from openai import AsyncOpenAI
from app.core.config import settings
from app.core.rag_utils import TextSplitter, SimpleVectorStore

logger = logging.getLogger(__name__)

class SecurityGuard:
    """
    Security sanitization for LLM inputs using prompt delimiting.
    """
    USER_DATA_TAG = "user_data"
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        if not text:
            return ""
        # Remove control characters (except newlines/tabs)
        text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch >= " ")
        # Strip any existing XML-like delimiter tags
        text = text.replace(f"<{SecurityGuard.USER_DATA_TAG}>", "")
        text = text.replace(f"</{SecurityGuard.USER_DATA_TAG}>", "")
        return text.strip()

class RAGConfig:
    """Configuration for RAG service"""
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    # Retrieval
    DEFAULT_K: int = 4
    MAX_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    # Generation
    MODEL_NAME: str = "gpt-4o-mini"
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 1000
    # Caching
    CACHE_TTL_SECONDS: int = 3600
    MAX_CACHE_SIZE: int = 100
    # Concurrent
    MAX_CONCURRENT_REQUESTS: int = 5

class VectorStoreCache:
    """LRU cache for loaded vector stores"""
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[SimpleVectorStore, datetime]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[SimpleVectorStore]:
        async with self._lock:
            if key in self._cache:
                store, timestamp = self._cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    return store
                else:
                    del self._cache[key]
            return None
    
    async def set(self, key: str, store: SimpleVectorStore):
        async with self._lock:
            if len(self._cache) >= self.max_size:
                # Evict oldest
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[key] = (store, datetime.now())

    async def invalidate(self, key: str):
         async with self._lock:
            if key in self._cache:
                del self._cache[key]

class RAGMetrics:
    """Track RAG service metrics (simplified)"""
    def __init__(self):
        self.total_queries = 0
        self.failed_queries = 0
    
    def record_query(self, success: bool):
        self.total_queries += 1
        if not success:
            self.failed_queries += 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "failed_queries": self.failed_queries,
        }

class EnhancedRAGService:
    """Enhanced RAG Service using native OpenAI and Numpy"""
    
    def __init__(self):
        self.config = RAGConfig()
        self.cache = VectorStoreCache(
            max_size=self.config.MAX_CACHE_SIZE,
            ttl_seconds=self.config.CACHE_TTL_SECONDS
        )
        self.metrics = RAGMetrics()
        self.semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_REQUESTS)
        
        self.api_key = settings.OPENAI_API_KEY
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            self.client = AsyncOpenAI(api_key=self.api_key)
            logger.info("RAG Service initialized (Native OpenAI)")
        else:
            self.client = None
            logger.warning("RAG Service DISABLED (No API Key)")

        self.text_splitter = TextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using OpenAI"""
        if not self.client:
            raise RuntimeError("RAG Service Disabled")
            
        # OpenAI batch limit is often 2048, but let's do smaller batches safely
        # model="text-embedding-3-small"
        
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def ingest_report(self, task_id: str, text_content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ingest report text into in-memory vector store"""
        if not self.enabled:
            return {"success": False, "error": "Disabled"}

        try:
            # Split
            text_content = SecurityGuard.sanitize_input(text_content)
            chunks = self.text_splitter.split_text(text_content)
            if not chunks:
                return {"success": False, "error": "No text chunks"}

            # Embed
            embeddings = await self._get_embeddings(chunks)

            # Store
            store = SimpleVectorStore()
            metadatas = [{"task_id": task_id, "chunk_index": i, **(metadata or {})} for i in range(len(chunks))]
            store.add_texts(chunks, embeddings, metadatas)

            await self.cache.set(task_id, store)
            
            return {"success": True, "num_chunks": len(chunks)}
        except Exception as e:
            logger.error(f"Ingest failed: {e}")
            return {"success": False, "error": str(e)}

    def ingest_report_sync(self, task_id: str, text_content: str, metadata: Optional[Dict[str, Any]] = None):
         """Sync wrapper for Celery"""
         return asyncio.run(self.ingest_report(task_id, text_content, metadata))

    async def chat_with_report(self, task_id: str, question: str, k: int = 4, include_sources: bool = False) -> Dict[str, Any]:
        """Chat with the report context"""
        async with self.semaphore:
            if not self.enabled:
                return {"success": False, "answer": "RAG Disabled", "error": "No API Key"}
            
            try:
                # 1. Get Store
                store = await self.cache.get(task_id)
                if not store:
                    return {"success": False, "error": "Report not processed or expired."}

                # 2. Embed Question
                q_embed = (await self._get_embeddings([question]))[0]

                # 3. Retrieve
                results = store.similarity_search_with_score(q_embed, k=k)
                
                # Filter by threshold
                relevant_docs = [
                    (doc, score) for doc, score in results 
                    if score >= self.config.SIMILARITY_THRESHOLD
                ]

                context_str = ""
                if not relevant_docs:
                     context_str = "No specific context found in the report."
                else:
                    context_str = "\n\n".join([d['content'] for d, s in relevant_docs])

                # 4. Generate Answer
                system_prompt = f"""You are a helpful data analyst. Answer the user question based ONLY on the context below.
If the answer is not in the context, say so.

Context:
{context_str}
"""
                response = await self.client.chat.completions.create(
                    model=self.config.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=self.config.TEMPERATURE,
                    max_tokens=self.config.MAX_TOKENS
                )

                answer = response.choices[0].message.content

                result = {
                    "success": True, 
                    "answer": answer,
                    "task_id": task_id
                }
                if include_sources:
                    result["sources"] = [{"content": d['content'], "score": s} for d, s in relevant_docs]
                
                self.metrics.record_query(True)
                return result

            except Exception as e:
                logger.error(f"Chat failed: {e}")
                self.metrics.record_query(False)
                return {"success": False, "error": str(e)}

_rag_service_instance = None

def get_rag_service():
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = EnhancedRAGService()
    return _rag_service_instance

rag_service = get_rag_service()