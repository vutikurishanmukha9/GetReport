import os
import logging
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache
from contextlib import asynccontextmanager

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGConfig:
    """Configuration for RAG service"""
    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]
    
    # Retrieval
    DEFAULT_K: int = 4
    MAX_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Generation
    MODEL_NAME: str = "gpt-4o-mini"  # More cost-effective
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 1000
    
    # Caching
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    MAX_CACHE_SIZE: int = 100
    
    # Performance
    BATCH_SIZE: int = 100
    MAX_CONCURRENT_REQUESTS: int = 5


class VectorStoreCache:
    """LRU cache for loaded vector stores"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[FAISS, datetime]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[FAISS]:
        """Get vector store from cache if valid"""
        async with self._lock:
            if key in self._cache:
                store, timestamp = self._cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    logger.debug(f"Cache hit for {key}")
                    return store
                else:
                    del self._cache[key]
                    logger.debug(f"Cache expired for {key}")
            return None
    
    async def set(self, key: str, store: FAISS):
        """Add vector store to cache"""
        async with self._lock:
            if len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
                logger.debug(f"Cache evicted {oldest_key}")
            
            self._cache[key] = (store, datetime.now())
            logger.debug(f"Cached {key}")
    
    async def invalidate(self, key: str):
        """Remove from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Invalidated cache for {key}")
    
    async def clear(self):
        """Clear all cache"""
        async with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")


class RAGMetrics:
    """Track RAG service metrics"""
    
    def __init__(self):
        self.total_ingestions = 0
        self.total_queries = 0
        self.failed_ingestions = 0
        self.failed_queries = 0
        self.total_chunks_created = 0
        self.avg_retrieval_time_ms = 0.0
        self.avg_generation_time_ms = 0.0
    
    def record_ingestion(self, success: bool, num_chunks: int = 0):
        self.total_ingestions += 1
        if success:
            self.total_chunks_created += num_chunks
        else:
            self.failed_ingestions += 1
    
    def record_query(self, success: bool, retrieval_ms: float = 0, generation_ms: float = 0):
        self.total_queries += 1
        if success:
            # Running average
            n = self.total_queries - self.failed_queries
            self.avg_retrieval_time_ms = (
                (self.avg_retrieval_time_ms * (n - 1) + retrieval_ms) / n
            )
            self.avg_generation_time_ms = (
                (self.avg_generation_time_ms * (n - 1) + generation_ms) / n
            )
        else:
            self.failed_queries += 1
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_ingestions": self.total_ingestions,
            "total_queries": self.total_queries,
            "failed_ingestions": self.failed_ingestions,
            "failed_queries": self.failed_queries,
            "total_chunks_created": self.total_chunks_created,
            "avg_retrieval_time_ms": round(self.avg_retrieval_time_ms, 2),
            "avg_generation_time_ms": round(self.avg_generation_time_ms, 2),
            "success_rate": round(
                (self.total_queries - self.failed_queries) / max(self.total_queries, 1) * 100, 2
            ),
        }


class EnhancedRAGService:
    """Enhanced RAG Service with caching, metrics, and improved error handling"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.vector_store_dir = Path(os.getcwd()) / "outputs" / "vector_stores"
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache and metrics
        self.cache = VectorStoreCache(
            max_size=self.config.MAX_CACHE_SIZE,
            ttl_seconds=self.config.CACHE_TTL_SECONDS
        )
        self.metrics = RAGMetrics()
        
        # Semaphore for rate limiting
        self.semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_REQUESTS)
        
        # Validate API Key
        self.api_key = settings.OPENAI_API_KEY
        self.enabled = bool(self.api_key)
        
        if not self.enabled:
            logger.warning("OPENAI_API_KEY is not set. RAG Service is DISABLED.")
        else:
            logger.info("EnhancedRAGService initialized (Enabled)")
    
    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Lazy-load embeddings model"""
        if not self.enabled:
             raise RuntimeError("RAG Service is disabled (No API Key)")
             
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                api_key=self.api_key,
                model="text-embedding-3-small",  # More efficient
                chunk_size=self.config.BATCH_SIZE
            )
        return self._embeddings
    
    @property
    def llm(self) -> ChatOpenAI:
        """Lazy-load LLM"""
        if not self.enabled:
             raise RuntimeError("RAG Service is disabled (No API Key)")

        if self._llm is None:
            self._llm = ChatOpenAI(
                model_name=self.config.MODEL_NAME,
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS,
                api_key=self.api_key,
                request_timeout=30,
            )
        return self._llm
    
    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Lazy-load text splitter"""
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                separators=self.config.SEPARATORS,
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                length_function=len,
                is_separator_regex=False,
            )
        return self._text_splitter
    
    def _get_index_path(self, task_id: str) -> Path:
        """Get path for vector store index"""
        return self.vector_store_dir / task_id
    
    async def _validate_text_content(self, text_content: str) -> Tuple[bool, Optional[str]]:
        """Validate input text content"""
        if not text_content or not text_content.strip():
            return False, "Text content is empty"
        
        if len(text_content) < 100:
            return False, "Text content is too short (minimum 100 characters)"
        
        if len(text_content) > 10_000_000:  # 10MB limit
            return False, "Text content is too large (maximum 10MB)"
        
        return True, None
    
    async def ingest_report(
        self,
        task_id: str,
        text_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Chunk text and build FAISS index with enhanced error handling and metrics.
        
        Args:
            task_id: Unique identifier for the task
            text_content: Text to ingest
            metadata: Optional metadata to attach to chunks
        
        Returns:
            Dict with ingestion results
        """
        start_time = datetime.now()
        
        try:
            if not self.enabled:
                logger.info(f"Skipping RAG ingestion for task {task_id} (Service Disabled)")
                return {
                    "success": False,
                    "error": "RAG Service Disabled (No API Key)",
                    "task_id": task_id
                }

            logger.info(f"Starting RAG ingestion for task {task_id}")
            
            # Validate input
            is_valid, error_msg = await self._validate_text_content(text_content)
            if not is_valid:
                logger.warning(f"Validation failed for task {task_id}: {error_msg}")
                self.metrics.record_ingestion(success=False)
                return {
                    "success": False,
                    "error": error_msg,
                    "task_id": task_id
                }
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text_content)
            
            if not chunks:
                logger.warning(f"No chunks generated for task {task_id}")
                self.metrics.record_ingestion(success=False)
                return {
                    "success": False,
                    "error": "No text chunks generated",
                    "task_id": task_id
                }
            
            logger.info(f"Generated {len(chunks)} chunks for task {task_id}")
            
            # Create documents with metadata
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "task_id": task_id,
                        "chunk_index": i,
                        "timestamp": datetime.now().isoformat(),
                        **(metadata or {})
                    }
                )
                for i, chunk in enumerate(chunks)
            ]
            
            # Build vector store
            logger.info(f"Building FAISS index for {len(documents)} documents")
            vector_store = await asyncio.to_thread(
                FAISS.from_documents,
                documents,
                self.embeddings
            )
            
            # Save to disk
            index_path = self._get_index_path(task_id)
            await asyncio.to_thread(vector_store.save_local, str(index_path))
            logger.info(f"FAISS index saved to {index_path}")
            
            # Update cache
            await self.cache.set(task_id, vector_store)
            
            # Record metrics
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.record_ingestion(success=True, num_chunks=len(chunks))
            
            return {
                "success": True,
                "task_id": task_id,
                "num_chunks": len(chunks),
                "index_path": str(index_path),
                "elapsed_ms": round(elapsed_ms, 2)
            }
        
        except Exception as e:
            logger.error(f"RAG ingestion failed for task {task_id}: {e}", exc_info=True)
            self.metrics.record_ingestion(success=False)
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }
    
    async def _load_vector_store(self, task_id: str) -> Optional[FAISS]:
        """Load vector store from cache or disk"""
        # Try cache first
        cached_store = await self.cache.get(task_id)
        if cached_store:
            return cached_store
        
        # Load from disk
        index_path = self._get_index_path(task_id)
        if not index_path.exists():
            logger.warning(f"Index not found for task {task_id}")
            return None
        
        try:
            vector_store = await asyncio.to_thread(
                FAISS.load_local,
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            await self.cache.set(task_id, vector_store)
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load vector store for task {task_id}: {e}")
            return None
    
    async def chat_with_report(
        self,
        task_id: str,
        question: str,
        k: Optional[int] = None,
        include_sources: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve context and answer question with enhanced features.
        
        Args:
            task_id: Task identifier
            question: User question
            k: Number of documents to retrieve (default: config.DEFAULT_K)
            include_sources: Whether to include source chunks in response
        
        Returns:
            Dict with answer and metadata
        """
        async with self.semaphore:
            return await self._chat_with_report_impl(task_id, question, k, include_sources)
    
    async def _chat_with_report_impl(
        self,
        task_id: str,
        question: str,
        k: Optional[int],
        include_sources: bool
    ) -> Dict[str, Any]:
        """Internal implementation of chat_with_report"""
        start_time = datetime.now()
        retrieval_time = None
        generation_time = None
        
        try:
            if not self.enabled:
                return {
                    "success": False,
                    "answer": "AI Chat is disabled because no OpenAI API Key was provided.",
                    "error": "Service Disabled",
                    "task_id": task_id
                }

            # Validate question
            if not question or not question.strip():
                return {
                    "success": False,
                    "error": "Question cannot be empty",
                    "task_id": task_id
                }
            
            # Load vector store
            vector_store = await self._load_vector_store(task_id)
            if not vector_store:
                self.metrics.record_query(success=False)
                return {
                    "success": False,
                    "error": "Report not found. Please ensure the analysis is complete.",
                    "task_id": task_id
                }
            
            # Retrieval
            retrieval_start = datetime.now()
            k = min(k or self.config.DEFAULT_K, self.config.MAX_K)
            
            docs = await asyncio.to_thread(
                vector_store.similarity_search_with_score,
                question,
                k=k
            )
            retrieval_time = (datetime.now() - retrieval_start).total_seconds() * 1000
            
            # Filter by similarity threshold
            docs = [(doc, score) for doc, score in docs if score >= self.config.SIMILARITY_THRESHOLD]
            
            if not docs:
                self.metrics.record_query(success=True, retrieval_ms=retrieval_time)
                return {
                    "success": True,
                    "answer": "I couldn't find any relevant information in the report to answer that question.",
                    "task_id": task_id,
                    "sources": [],
                    "retrieval_time_ms": round(retrieval_time, 2)
                }
            
            # Prepare context
            context_text = "\n\n".join([doc.page_content for doc, _ in docs])
            
            # Generation
            generation_start = datetime.now()
            
            prompt = ChatPromptTemplate.from_template(
                """You are a helpful data analyst assistant. Answer the question based ONLY on the following context from the analysis report.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, concise answer based on the context
- If the context doesn't contain enough information, say "I don't see that information in the report."
- Do not make up information or use external knowledge
- If relevant, cite specific details from the context

Answer:"""
            )
            
            chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            answer = await chain.ainvoke({"context": context_text, "question": question})
            generation_time = (datetime.now() - generation_start).total_seconds() * 1000
            
            # Prepare response
            response = {
                "success": True,
                "answer": answer.strip(),
                "task_id": task_id,
                "num_sources": len(docs),
                "retrieval_time_ms": round(retrieval_time, 2),
                "generation_time_ms": round(generation_time, 2),
                "total_time_ms": round((datetime.now() - start_time).total_seconds() * 1000, 2)
            }
            
            if include_sources:
                response["sources"] = [
                    {
                        "content": doc.page_content,
                        "score": float(score),
                        "metadata": doc.metadata
                    }
                    for doc, score in docs
                ]
            
            # Record metrics
            self.metrics.record_query(
                success=True,
                retrieval_ms=retrieval_time,
                generation_ms=generation_time
            )
            
            return response
        
        except Exception as e:
            logger.error(f"RAG chat failed for task {task_id}: {e}", exc_info=True)
            self.metrics.record_query(success=False)
            return {
                "success": False,
                "error": f"Error processing question: {str(e)}",
                "task_id": task_id
            }
    
    async def delete_index(self, task_id: str) -> bool:
        """Delete vector store index"""
        try:
            index_path = self._get_index_path(task_id)
            if index_path.exists():
                await asyncio.to_thread(
                    lambda: [f.unlink() for f in index_path.glob("*")]
                )
                index_path.rmdir()
                logger.info(f"Deleted index for task {task_id}")
            
            await self.cache.invalidate(task_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete index for task {task_id}: {e}")
            return False
    
    async def list_indices(self) -> List[Dict[str, Any]]:
        """List all available indices"""
        try:
            indices = []
            for path in self.vector_store_dir.iterdir():
                if path.is_dir():
                    indices.append({
                        "task_id": path.name,
                        "path": str(path),
                        "created": datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
                        "size_mb": sum(f.stat().st_size for f in path.glob("*")) / (1024 * 1024)
                    })
            return indices
        except Exception as e:
            logger.error(f"Failed to list indices: {e}")
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return self.metrics.get_stats()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            # Test embeddings
            test_embedding = await asyncio.to_thread(
                self.embeddings.embed_query,
                "test"
            )
            
            return {
                "status": "healthy",
                "embeddings": "ok" if test_embedding else "error",
                "api_key_set": bool(settings.OPENAI_API_KEY),
                "cache_size": len(self.cache._cache),
                "metrics": self.get_metrics()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Singleton instance
_rag_service_instance: Optional[EnhancedRAGService] = None


def get_rag_service() -> EnhancedRAGService:
    """Get or create RAG service singleton"""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = EnhancedRAGService()
    return _rag_service_instance


# Backward compatibility
rag_service = get_rag_service()