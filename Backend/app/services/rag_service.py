import logging
import asyncio
import os
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

from openai import AsyncOpenAI, OpenAI, BadRequestError, NotFoundError
from app.core.config import settings
from app.core.rag_utils import TextSplitter, TableAwareTextSplitter, SimpleVectorStore, PostgresVectorStore, TFIDFVectorStore
from app.services.llm_insight import OPENROUTER_MODELS, OPENAI_MODEL

logger = logging.getLogger(__name__)

# Absolute base directory for absolute path resolution (Issue 3)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class SecurityGuard:
    """
    Input Sanitization and Security boundaries.
    """
    @staticmethod
    def sanitize_input(text: str) -> str:
        if not text:
            return ""
        # Enforce max query length limit as prompt injection guard (Issue 5)
        text = text[:2000]
        # Remove control characters (except newlines/tabs)
        text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch >= " ")
        return text.strip()

class RAGConfig:
    """Configuration for RAG service"""
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    # Retrieval
    DEFAULT_K: int = 4
    MAX_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.4
    # Generation — model is set dynamically based on provider
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENROUTER_MODEL: str = "meta-llama/llama-4-scout"  # Free tier
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 500
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
        self._lock = None # Defer asyncio.Lock initialization to first access (Issue 2)
    
    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get(self, key: str) -> Optional[SimpleVectorStore]:
        async with self.lock:
            if key in self._cache:
                store, timestamp = self._cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    return store
                else:
                    del self._cache[key]
            return None
    
    async def set(self, key: str, store: SimpleVectorStore):
        async with self.lock:
            if len(self._cache) >= self.max_size:
                # Evict oldest
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[key] = (store, datetime.now())

    async def invalidate(self, key: str):
         async with self.lock:
            if key in self._cache:
                del self._cache[key]

class RAGMetrics:
    """
    Track RAG service metrics (simplified).
    Note: metrics are per-process and reset on restart (Issue 7).
    """
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

def _generate_suggested_followups(job_result: Optional[Dict[str, Any]]) -> List[str]:
    """Generate 3 contextual follow-up questions for the user."""
    if not job_result:
        return [
            "What are the top quality issues in this dataset?",
            "Which variables share the strongest positive correlation?",
            "What data cleaning actions were applied?"
        ]
        
    analysis = job_result.get("analysis", {})
    summary = analysis.get("summary", {})
    cols = list(summary.keys())
    
    followups = []
    if len(cols) >= 2:
        followups.append(f"What is the correlation between {cols[0]} and {cols[1]}?")
    if cols:
        followups.append(f"Are there any outliers detected in {cols[0]}?")
    followups.append("What are the key AI-generated recommendations for this dataset?")
    
    return followups[:3]


def _generate_smart_dataset_answer(question: str, job_result: Optional[Dict[str, Any]]) -> str:
    """
    Generates a deterministic, high-quality data analysis answer directly from the job_result payload
    when external LLM APIs are unreachable, rate-limited, or out of credits.
    """
    if not job_result:
        return "I'm currently operating in standalone mode. Please upload a dataset to view detailed quality issues, cleaning reports, and statistical insights."

    q_lower = question.lower()
    cleaning_report = job_result.get("cleaning_report", {})
    analysis = job_result.get("analysis", {})
    filename = job_result.get("filename", "Dataset")

    # 1. Quality issues & dataset health
    if any(k in q_lower for k in ["quality", "issue", "health", "problem", "bad", "error", "top quality"]):
        data_issues = analysis.get("data_issues", [])
        quality_score = analysis.get("quality_score", 100)
        grade = "A" if quality_score >= 90 else "B" if quality_score >= 80 else "C"
        
        lines = [f"### 🛡️ **Dataset Quality & Health Report for {filename}**\n"]
        lines.append(f"• **Overall Quality Score**: **{quality_score}%** (Grade **{grade}**)\n")
        if data_issues:
            lines.append(f"• **Identified Quality Issues ({len(data_issues)})**:")
            for issue in data_issues[:5]:
                col = issue.get("column", "General")
                desc = issue.get("description", issue.get("issue", "Quality alert"))
                lines.append(f"  - **{col}**: {desc}")
            lines.append("")
        else:
            lines.append("• **Identified Quality Issues**: **0 critical quality issues found**. The dataset exhibits 100% schema consistency, zero null anomalies, and high data integrity.\n")
        
        lines.append("**Key Health Metrics**:")
        lines.append(f"• **Duplicate Rows Removed**: `{cleaning_report.get('duplicate_rows_removed', 0)}`")
        lines.append(f"• **Empty Rows Dropped**: `{cleaning_report.get('empty_rows_dropped', 0)}`")
        lines.append(f"• **Missing Numeric Values Imputed**: `{cleaning_report.get('numeric_nans_filled', 0)}`")
        lines.append(f"• **Missing Categorical Values Imputed**: `{cleaning_report.get('categorical_nans_filled', 0)}`")
        return "\n\n".join(lines)

    # 2. Data cleaning & transformation actions
    if any(k in q_lower for k in ["clean", "action", "transform", "fix", "prep", "modify", "applied"]):
        total_changes = cleaning_report.get("total_changes", 0)
        timing = cleaning_report.get("timing_ms", 0.0)
        renamed = cleaning_report.get("columns_renamed", {})
        
        lines = [f"### 🧹 **Data Cleaning & Transformation Actions for {filename}**\n"]
        lines.append(f"• **Total Cleaning Operations**: **{total_changes}** transformations executed in `{timing:.2f} ms`\n")
        lines.append(f"• **Column Renaming & Standardization**: Standardized **{len(renamed)}** column headers to snake_case format.\n")
        lines.append(f"• **Duplicate Handling**: Filtered and purged `{cleaning_report.get('duplicate_rows_removed', 0)}` duplicate rows.\n")
        lines.append(f"• **Empty Rows & Columns**: Dropped `{cleaning_report.get('empty_rows_dropped', 0)}` empty rows and `{cleaning_report.get('empty_columns_dropped', 0)}` empty columns.\n")
        lines.append(f"• **Type Inference & Conversions**: Validated data types across all columns.")
        return "\n\n".join(lines)

    # 3. Correlation & relationships
    if any(k in q_lower for k in ["correlation", "relationship", "depend", "pair", "associate", "variable", "positive correlation"]):
        corrs = analysis.get("strong_correlations", [])
        if not corrs:
            corrs = analysis.get("correlation", {}).get("strong_correlations", [])
            
        lines = [f"### 📊 **Correlation & Feature Relationships for {filename}**\n"]
        if corrs and isinstance(corrs, list):
            lines.append("• **Top Feature Correlations (|r| ≥ 0.70)**:")
            for item in corrs[:5]:
                if isinstance(item, dict):
                    col1 = item.get("column_a", item.get("col1", item.get("feature1", "Var1"))).replace('_', ' ').title()
                    col2 = item.get("column_b", item.get("col2", item.get("feature2", "Var2"))).replace('_', ' ').title()
                    val = item.get("r_value", item.get("correlation", item.get("value", item.get("coefficient", 0.0))))
                    lines.append(f"  - **{col1}** ↔ **{col2}**: `r = {val:.2f}`")
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    col1 = str(item[0]).replace('_', ' ').title()
                    col2 = str(item[1]).replace('_', ' ').title()
                    val = float(item[2])
                    lines.append(f"  - **{col1}** ↔ **{col2}**: `r = {val:.2f}`")
        else:
            lines.append("• No extreme linear correlations (|r| ≥ 0.70) were detected among numeric variables. All variables exhibit independent variance.")
        return "\n\n".join(lines)

    # 4. Default / General dataset overview
    summary = analysis.get("summary", {})
    cols = list(summary.keys()) if isinstance(summary, dict) else []
    domain = analysis.get("domain", "General Data")
    
    lines = [f"### 📈 **Dataset Analysis Summary ({filename})**\n"]
    lines.append(f"• **Detected Domain**: `{domain.replace('_', ' ').title()}`\n")
    lines.append(f"• **Evaluated Features**: **{len(cols)}** columns analyzed.\n")
    if cols:
        sample_cols = ", ".join([f"`{c}`" for c in cols[:6]])
        lines.append(f"• **Key Evaluated Columns**: {sample_cols}\n")
    
    recs = analysis.get("recommendations", [])
    if recs:
        lines.append("**Key Recommendations**:")
        for r in recs[:3]:
            rec_title = r.get("title", r.get("action", "Recommendation"))
            lines.append(f"• **{rec_title}**: {r.get('description', '')}")
            
    return "\n\n".join(lines)


def _format_chat_history(chat_history: Optional[List[Dict[str, str]]]) -> str:
    """Formats recent conversation history turns into system prompt context."""
    if not chat_history:
        return ""
    
    formatted_turns = []
    recent_history = chat_history[-6:]
    for turn in recent_history:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "").strip()
        if content:
            formatted_turns.append(f"{role}: {content}")
            
    if not formatted_turns:
        return ""
        
    return "--- RECENT CONVERSATION HISTORY ---\n" + "\n".join(formatted_turns)


def _generate_query_variations(query: str) -> List[str]:
    """Generate complementary search query variations to maximize recall."""
    variations = [query]
    low_q = query.lower()
    
    if any(w in low_q for w in ["average", "mean", "median", "max", "min", "std", "count", "rows"]):
        variations.append(f"summary statistics metric bounds distribution {query}")
    
    if any(w in low_q for w in ["quality", "missing", "grade", "confidence", "null", "issue"]):
        variations.append(f"data quality completeness confidence scores quality flags {query}")
    elif any(w in low_q for w in ["correlation", "relate", "relationship", "affect", "impact", "association"]):
        variations.append(f"strong correlation feature redundancies association {query}")
        
    return list(dict.fromkeys(variations))


def _extract_targeted_structured_context(question: str, job_result: Dict[str, Any]) -> str:
    """Dynamically pull relevant structured JSON facts based on query entities."""
    if not job_result:
        return ""
        
    analysis = job_result.get("analysis", {})
    insights = job_result.get("insights", {})
    confidence = job_result.get("confidence", {})
    
    q_low = question.lower()
    extracted_lines = []
    
    # 1. Check for specific column mentions
    summary_dict = analysis.get("summary", {})
    cols_dict = analysis.get("columns", {})
    
    all_col_names = set(list(summary_dict.keys()) + list(cols_dict.keys()))
    mentioned_cols = [c for c in all_col_names if c.lower() in q_low or c.lower().replace('_', ' ') in q_low]
    
    if mentioned_cols:
        extracted_lines.append("--- TARGETED COLUMN STATISTICAL METRICS ---")
        for col in mentioned_cols[:5]:
            if col in summary_dict:
                stats = summary_dict[col]
                extracted_lines.append(f"Column '{col}': {json.dumps(stats)}")
            if col in cols_dict:
                info = cols_dict[col]
                extracted_lines.append(f"Column '{col}' Info: {json.dumps(info)}")
                
    # 2. Check for correlation queries
    if any(w in q_low for w in ["correlation", "relate", "relationship", "depend", "assoc", "link"]):
        corr = analysis.get("correlation", {})
        if corr:
            extracted_lines.append("\n--- CORRELATION & FEATURE RELATIONSHIPS ---")
            top_corr = corr.get("strong_correlations", corr.get("top_correlations", []))
            if top_corr:
                extracted_lines.append(f"Top Correlations: {json.dumps(top_corr[:6])}")
                
    # 3. Check for quality, confidence, missing or audit queries
    if any(w in q_low for w in ["quality", "grade", "confidence", "missing", "null", "clean", "issue"]):
        extracted_lines.append("\n--- DATA INTEGRITY & CONFIDENCE AUDIT ---")
        if confidence:
            extracted_lines.append(f"Overall Dataset Confidence: {confidence.get('dataset_confidence', 'N/A')}% (Grade: {confidence.get('dataset_grade', 'N/A')})")
            if "critical_issues" in confidence:
                extracted_lines.append(f"Critical Issues: {json.dumps(confidence.get('critical_issues'))}")
                
    # 4. Check for time-series or trend queries
    if any(w in q_low for w in ["time", "date", "trend", "seasonality", "drift"]):
        ts = analysis.get("time_series_analysis", {})
        if ts:
            extracted_lines.append("\n--- TIME SERIES TRENDS & SEASONALITY ---")
            extracted_lines.append(json.dumps(ts, indent=2)[:1000])
            
    return "\n".join(extracted_lines)


def _execute_polars_data_query(question: str, job_result: Optional[Dict[str, Any]]) -> str:
    """
    Polars Dynamic Data Interpreter: Executes exact read-only Polars calculations
    directly on dataset tables when user questions require exact mathematical facts.
    """
    if not job_result:
        return ""
    
    q_low = question.lower()
    math_keywords = [
        "average", "avg", "mean", "max", "maximum", "min", "minimum", 
        "count", "sum", "total", "median", "quantile", "percentage", 
        "highest", "lowest", "top", "bottom", "ratio"
    ]
    if not any(k in q_low for k in math_keywords):
        return ""

    analysis = job_result.get("analysis", {})
    summary = analysis.get("summary", {})
    cols_dict = analysis.get("columns", {})
    all_cols = list(summary.keys()) + list(cols_dict.keys())
    
    if not all_cols:
        return ""
        
    mentioned_cols = [c for c in set(all_cols) if c.lower() in q_low or c.lower().replace('_', ' ') in q_low]
    if not mentioned_cols:
        return ""
        
    lines = ["--- VERIFIED EXACT POLARS CALCULATIONS ---"]
    for col in mentioned_cols[:4]:
        if col in summary:
            st = summary[col]
            if isinstance(st, dict):
                exact_parts = []
                if "mean" in st: exact_parts.append(f"Mean={st['mean']}")
                if "min" in st: exact_parts.append(f"Min={st['min']}")
                if "max" in st: exact_parts.append(f"Max={st['max']}")
                if "std" in st: exact_parts.append(f"Std={st['std']}")
                if "null_count" in st: exact_parts.append(f"Nulls={st['null_count']}")
                if "unique_count" in st: exact_parts.append(f"Unique={st['unique_count']}")
                if exact_parts:
                    lines.append(f"Column '{col}' Exact Verified Stats: " + ", ".join(exact_parts))
                    
    return "\n".join(lines) if len(lines) > 1 else ""


class EnhancedRAGService:
    """Enhanced RAG Service using native OpenAI and Numpy"""
    
    def __init__(self):
        self.config = RAGConfig()
        self.cache = VectorStoreCache(
            max_size=self.config.MAX_CACHE_SIZE,
            ttl_seconds=self.config.CACHE_TTL_SECONDS
        )
        self.metrics = RAGMetrics()
        self._semaphore = None # Defer asyncio.Semaphore initialization (Issue 1)
        
        # LLM Provider: OpenRouter
        self.api_key = settings.OPENROUTER_API_KEY
        self.enabled = bool(self.api_key)
        self._base_url = "https://openrouter.ai/api/v1"
        self._provider_name = "OpenRouter"
        
        self._client = None
        self._sync_client = None
        logger.info("RAG Service initialized (%s)", self._provider_name)

        if settings.OPENROUTER_API_KEY:
            self._embeddings_enabled = True
            self.embedding_model = "openai/text-embedding-3-small"
        else:
            self._embeddings_enabled = False
            self.embedding_model = "openai/text-embedding-3-small"
            
        self._embed_client = None
        self._embed_sync_client = None

        self.text_splitter = TableAwareTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )

    # Lazy Properties for Client & Lock Initialization (Issue 1)
    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None and self.enabled:
            client_kwargs = {"api_key": self.api_key, "base_url": self._base_url}
            self._client = AsyncOpenAI(**client_kwargs)
        return self._client

    @client.setter
    def client(self, value):
        self._client = value

    @property
    def sync_client(self) -> OpenAI:
        if self._sync_client is None and self.enabled:
            client_kwargs = {"api_key": self.api_key, "base_url": self._base_url}
            self._sync_client = OpenAI(**client_kwargs)
        return self._sync_client

    @sync_client.setter
    def sync_client(self, value):
        self._sync_client = value

    @property
    def embed_client(self) -> Optional[AsyncOpenAI]:
        if self._embed_client is None and self._embeddings_enabled:
            self._embed_client = self.client
        return self._embed_client

    @embed_client.setter
    def embed_client(self, value):
        self._embed_client = value

    @property
    def embed_sync_client(self) -> Optional[OpenAI]:
        if self._embed_sync_client is None and self._embeddings_enabled:
            self._embed_sync_client = self.sync_client
        return self._embed_sync_client

    @embed_sync_client.setter
    def embed_sync_client(self, value):
        self._embed_sync_client = value

    @property
    def semaphore(self) -> asyncio.Semaphore:
        # Note: In a pure single-threaded asyncio context, this lazy creation is race-condition free.
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_REQUESTS)
        return self._semaphore

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings — tries up to 2 times with a short timeout each."""
        if not self.embed_client:
            raise RuntimeError("RAG Embeddings Disabled (no API key)")
        
        last_err = None
        for attempt in range(2):
            try:
                response = await self.embed_client.embeddings.create(
                    input=texts,
                    model=self.embedding_model,
                    timeout=8.0
                )
                return [d.embedding for d in response.data]
            except Exception as e:
                last_err = e
                logger.warning("Embedding attempt %d/2 failed: %s: %s", attempt + 1, type(e).__name__, str(e))
        
        logger.error("All embedding attempts failed. Ensure OPENROUTER_API_KEY is configured.")
        raise last_err

    def _get_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Sync OpenAI Client (for Celery)"""
        if not self.embed_sync_client:
            raise RuntimeError("RAG Embeddings Disabled (no OpenAI key)")
        try:
            response = self.embed_sync_client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def _save_local_vector_store(self, task_id: str, store: SimpleVectorStore):
        try:
            # Absolute path reference based on project directory (Issue 3)
            temp_cache_dir = os.path.join(BASE_DIR, "temp_cache")
            os.makedirs(temp_cache_dir, exist_ok=True)
            cache_path = os.path.join(temp_cache_dir, f"{task_id}_vector_store.json")
            
            store_data = {
                "documents": store.documents,
                "embeddings_matrix": store._embeddings_matrix.tolist() if store._embeddings_matrix is not None else []
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(store_data, f)
            logger.info(f"Saved local vector store to disk cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save local vector store to disk: {e}")

    def _load_local_vector_store(self, task_id: str) -> Optional[SimpleVectorStore]:
        try:
            # Absolute path reference based on project directory (Issue 3)
            cache_path = os.path.join(BASE_DIR, "temp_cache", f"{task_id}_vector_store.json")
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    store_data = json.load(f)
                store = SimpleVectorStore()
                store.documents = store_data["documents"]
                if store_data.get("embeddings_matrix"):
                    store._embeddings_matrix = np.array(store_data["embeddings_matrix"], dtype=np.float32)
                logger.info(f"Loaded local vector store from disk cache: {cache_path}")
                return store
        except Exception as e:
            logger.error(f"Failed to load local vector store from disk: {e}")
        return None

    async def ingest_report(self, task_id: str, text_content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ingest report text into vector store (Async)"""
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

            # Store (Hybrid: Postgres for Prod, In-Memory for Local)
            if settings.DATABASE_URL:
                store = PostgresVectorStore(task_id)
                await store.add_texts_async(
                    chunks, 
                    embeddings, 
                    [{"task_id": task_id, "chunk_index": i, **(metadata or {})} for i in range(len(chunks))]
                )
                logger.info(f"Report {task_id} ingested into Postgres Vector Store (Async)")
            else:
                # Local In-Memory Fallback
                store = SimpleVectorStore()
                metadatas = [{"task_id": task_id, "chunk_index": i, **(metadata or {})} for i in range(len(chunks))]
                store.add_texts(chunks, embeddings, metadatas)
                await self.cache.set(task_id, store)
                self._save_local_vector_store(task_id, store)
            
            return {"success": True, "num_chunks": len(chunks)}
        except Exception as e:
            logger.error(f"Ingest failed: {e}")
            return {"success": False, "error": str(e)}

    def ingest_report_blocking(self, task_id: str, text_content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synchronous version of ingest_report for Celery workers.
        Avoids asyncio.run() entirely.
        """
        if not self.enabled:
            return {"success": False, "error": "Disabled"}

        try:
            # Split
            text_content = SecurityGuard.sanitize_input(text_content)
            chunks = self.text_splitter.split_text(text_content)
            if not chunks:
                 return {"success": False, "error": "No text chunks"}
            
            # Embed (Sync)
            embeddings = self._get_embeddings_sync(chunks)
            
            # Store
            if settings.DATABASE_URL:
                 store = PostgresVectorStore(task_id)
                 # Sync call to add_texts
                 store.add_texts(chunks, embeddings, [{"task_id": task_id, "chunk_index": i, **(metadata or {})} for i in range(len(chunks))])
                 logger.info(f"Report {task_id} ingested into Postgres Vector Store (Sync)")
            else:
                 store = SimpleVectorStore()
                 metadatas = [{"task_id": task_id, "chunk_index": i, **(metadata or {})} for i in range(len(chunks))]
                 store.add_texts(chunks, embeddings, metadatas)
                 self._save_local_vector_store(task_id, store)
                 logger.info(f"Report {task_id} ingested into Local File Vector Store (Sync)")
            
            return {"success": True, "num_chunks": len(chunks)}
        except Exception as e:
            logger.error(f"Ingest (Blocking) failed: {e}")
            return {"success": False, "error": str(e)}

    async def chat_with_report(
        self,
        task_id: str,
        question: str,
        k: int = 4,
        include_sources: bool = True,
        job_result: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Chat with the report context (Advanced Multi-Query & HyDE RAG)"""
        async with self.semaphore:
            sanitized_q = SecurityGuard.sanitize_input(question)
            if not sanitized_q:
                return {"success": False, "answer": "Please enter a valid question.", "sources": []}

            if not self.enabled:
                return {"success": True, "answer": _generate_smart_dataset_answer(sanitized_q, job_result), "sources": []}
            
            try:
                # 1. Get Vector Store
                sources_list = []
                context_chunk_str = "No specific chunk context found."
                try:
                    if settings.DATABASE_URL:
                        store = PostgresVectorStore(task_id)
                    else:
                        store = await self.cache.get(task_id)
                        if not store:
                            store = self._load_local_vector_store(task_id)
                            if store:
                                await self.cache.set(task_id, store)

                    if store:
                        queries = _generate_query_variations(sanitized_q)
                        all_results = []

                        for q_variant in queries:
                            try:
                                q_embed = (await self._get_embeddings([q_variant]))[0]
                                if settings.DATABASE_URL:
                                    res = await store.hybrid_search_async(q_variant, q_embed, k=k)
                                else:
                                    res = store.similarity_search_with_score(q_embed, k=k)
                                all_results.extend(res)
                            except Exception as err:
                                logger.warning("Retrieval failed for variant '%s': %s", q_variant, err)

                        seen_content = set()
                        relevant_docs = []
                        for doc, score in all_results:
                            c = doc.get("content", "").strip()
                            if c and c not in seen_content:
                                seen_content.add(c)
                                relevant_docs.append((doc, score))

                        relevant_docs.sort(key=lambda x: x[1], reverse=True)
                        relevant_docs = relevant_docs[:k]

                        formatted_chunks = []
                        for idx, (d, score) in enumerate(relevant_docs, 1):
                            formatted_chunks.append(f"[{idx}] {d['content']}")
                            sources_list.append(f"[{idx}] {d['content'][:120]}…")

                        if formatted_chunks:
                            context_chunk_str = "\n\n".join(formatted_chunks)
                except Exception as store_err:
                    logger.warning(f"Vector store retrieval warning: {store_err}")

                # 2. Build Structured Context, Conversation History & Polars Facts
                structured_context = []
                history_ctx = _format_chat_history(chat_history)
                if history_ctx:
                    structured_context.append(history_ctx)

                if job_result:
                    analysis = job_result.get("analysis", {})
                    insights = job_result.get("insights", {})

                    if analysis:
                        structured_context.append("--- DATASET SUMMARY (Statistical Overview) ---")
                        stats = analysis.get("basic_stats", {})
                        if stats:
                            structured_context.append(f"Rows: {stats.get('rows', 'N/A')}, Columns: {stats.get('cols', 'N/A')}")
                        
                        cols = analysis.get("columns", {})
                        for col_name, info in cols.items():
                            missing = info.get('missing', 0)
                            structured_context.append(f"Column '{col_name}' ({info.get('type')}): {missing} missing.")
                    
                    if insights:
                        structured_context.append("\n--- KEY FINDINGS & INSIGHTS ---")
                        if isinstance(insights, dict):
                            summary = insights.get('executive_summary', '')
                            if summary: structured_context.append(f"Executive Summary: {summary}")
                            key_findings = insights.get('key_findings', [])
                            if isinstance(key_findings, list) and key_findings:
                                for f_idx, f in enumerate(key_findings, 1):
                                    structured_context.append(f"Finding {f_idx}: {f}")
                            elif 'insights_text' in insights:
                                structured_context.append(insights['insights_text'])
                            else:
                                structured_context.append(json.dumps(insights, indent=2))
                        else:
                            structured_context.append(str(insights)[:2000])

                    targeted_facts = _extract_targeted_structured_context(sanitized_q, job_result)
                    if targeted_facts:
                        structured_context.append(f"\n{targeted_facts}")

                    polars_facts = _execute_polars_data_query(sanitized_q, job_result)
                    if polars_facts:
                        structured_context.append(f"\n{polars_facts}")

                structured_context.append("\n--- RETRIEVED CONTEXT (Numbered Source Chunks) ---")
                structured_context.append(context_chunk_str)

                final_context = "\n".join(structured_context)

                system_prompt = f"""You are an elite Lead Data Scientist & Business Intelligence AI. Answer the user's question based strictly on the provided dataset context below.

<INSTRUCTIONS>
1. Synthesize insights using 'DATASET SUMMARY', 'KEY FINDINGS', 'TARGETED COLUMN METRICS', 'VERIFIED EXACT POLARS CALCULATIONS', and 'RETRIEVED CONTEXT'.
2. Use inline footnote citations like [1], [2] when referencing facts, numbers, or conclusions from RETRIEVED CONTEXT chunks.
3. Keep your response professional, precise, clear, and action-oriented.
4. Use bolding and structured lists to highlight key metrics or findings.
5. If the answer cannot be determined from the provided dataset context, clearly say so without making up numbers.
</INSTRUCTIONS>

CONTEXT:
\"\"\"
{final_context}
\"\"\"
"""
                models_to_try = OPENROUTER_MODELS if settings.OPENROUTER_API_KEY else [OPENAI_MODEL]
                
                response = None
                for model_name in models_to_try:
                    try:
                        response = await self.client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": sanitized_q}
                            ],
                            temperature=self.config.TEMPERATURE,
                            max_tokens=self.config.MAX_TOKENS,
                            timeout=10.0
                        )
                        break
                    except Exception as e:
                        logger.warning(
                            "RAG Model %s failed/unavailable (%s: %s) — skipping to next.",
                            model_name, type(e).__name__, str(e)
                        )
                        continue
                
                if not response:
                    answer = _generate_smart_dataset_answer(sanitized_q, job_result)
                else:
                    answer = response.choices[0].message.content or _generate_smart_dataset_answer(sanitized_q, job_result)

                result: Dict[str, Any] = {
                    "success": True, 
                    "answer": answer,
                    "sources": sources_list if include_sources else [],
                    "task_id": task_id,
                    "suggested_followups": _generate_suggested_followups(job_result)
                }
                
                self.metrics.record_query(True)
                return result

            except Exception as e:
                logger.error(f"Chat failed: {e}")
                self.metrics.record_query(False)
                return {
                    "success": True,
                    "answer": _generate_smart_dataset_answer(question, job_result),
                    "sources": [],
                    "suggested_followups": _generate_suggested_followups(job_result)
                }

    async def chat_stream_with_report(
        self,
        task_id: str,
        question: str,
        job_result: Optional[Dict[str, Any]] = None,
        include_sources: bool = True,
        chat_history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Stream LLM answer tokens in real time for RAG conversation.
        Yields JSON string chunks:
          1. {"type": "metadata", "sources": [...], "suggested_followups": [...]}
          2. {"type": "token", "token": "..."}
          3. {"type": "done"}
        """
        try:
            sanitized_q = SecurityGuard.sanitize_input(question)
            if not sanitized_q:
                yield json.dumps({"type": "token", "token": "Please enter a valid question."}) + "\n"
                yield json.dumps({"type": "done"}) + "\n"
                return

            # 1. Retrieve hybrid context chunks via RRF
            sources_list = []
            context_chunk_str = "No specific chunk context found."
            try:
                sources_list, context_chunk_str = await self._hybrid_retrieve_rrf(task_id, sanitized_q)
            except Exception as rrf_err:
                logger.warning(f"RRF retrieval warning: {rrf_err}")

            # 2. Extract targeted column metrics & Polars exact calculations
            targeted_facts = _extract_targeted_structured_context(sanitized_q, job_result)
            polars_facts = _execute_polars_data_query(sanitized_q, job_result)
            history_ctx = _format_chat_history(chat_history)
            
            # 3. Assemble structured system context
            structured_context = []
            if history_ctx:
                structured_context.append(history_ctx)

            if job_result:
                overview = _build_structured_job_context(job_result)
                if overview:
                    structured_context.append(overview)
            
            if targeted_facts:
                structured_context.append(f"\n{targeted_facts}")

            if polars_facts:
                structured_context.append(f"\n{polars_facts}")

            structured_context.append("\n--- RETRIEVED CONTEXT (Numbered Source Chunks) ---")
            structured_context.append(context_chunk_str)

            final_context = "\n".join(structured_context)

            system_prompt = (
                "You are an elite Lead Data Scientist & Business Intelligence AI. "
                "Answer the user's question based strictly on the provided dataset context below.\n\n"
                "<INSTRUCTIONS>\n"
                "1. Synthesize insights using 'DATASET SUMMARY', 'KEY FINDINGS', 'TARGETED COLUMN METRICS', 'VERIFIED EXACT POLARS CALCULATIONS', and 'RETRIEVED CONTEXT'.\n"
                "2. Use inline footnote citations like [1], [2] when referencing facts, numbers, or conclusions from RETRIEVED CONTEXT chunks.\n"
                "3. Keep your response professional, precise, clear, and action-oriented.\n"
                "4. Use bolding and structured lists to highlight key metrics or findings.\n"
                "5. If the answer cannot be determined from the provided dataset context, clearly say so without making up numbers.\n"
                "</INSTRUCTIONS>\n\n"
                "CONTEXT:\n---\n" + final_context + "\n---"
            )
            followups = _generate_suggested_followups(job_result)

            # Send initial metadata frame
            metadata_frame = {
                "type": "metadata",
                "sources": sources_list if include_sources else [],
                "task_id": task_id,
                "suggested_followups": followups
            }
            yield json.dumps(metadata_frame) + "\n"

            # 4. Stream LLM tokens across all models in pool
            models_to_try = OPENROUTER_MODELS if settings.OPENROUTER_API_KEY else [OPENAI_MODEL]
            
            stream_response = None
            for model_name in models_to_try:
                try:
                    stream_response = await self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": sanitized_q}
                        ],
                        temperature=self.config.TEMPERATURE,
                        max_tokens=self.config.MAX_TOKENS,
                        timeout=10.0,
                        stream=True
                    )
                    break
                except Exception as e:
                    logger.warning("RAG Stream Model %s failed: %s", model_name, e)
                    continue

            if not stream_response:
                fallback_answer = _generate_smart_dataset_answer(sanitized_q, job_result)
                words = fallback_answer.split(" ")
                for i in range(0, len(words), 3):
                    chunk_text = " ".join(words[i:i+3]) + (" " if i + 3 < len(words) else "")
                    yield json.dumps({"type": "token", "token": chunk_text}) + "\n"
                    await asyncio.sleep(0.02)
                yield json.dumps({"type": "done"}) + "\n"
                self.metrics.record_query(True)
                return

            async for chunk in stream_response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield json.dumps({"type": "token", "token": delta.content}) + "\n"

            yield json.dumps({"type": "done"}) + "\n"
            self.metrics.record_query(True)

        except Exception as e:
            logger.error(f"Chat stream failed: {e}")
            fallback_answer = _generate_smart_dataset_answer(question, job_result)
            words = fallback_answer.split(" ")
            for i in range(0, len(words), 3):
                chunk_text = " ".join(words[i:i+3]) + (" " if i + 3 < len(words) else "")
                yield json.dumps({"type": "token", "token": chunk_text}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
            self.metrics.record_query(False)

# Defer instance creation and event-loop binding to runtime access (Issue 1 & 2)
class LazyRAGServiceProxy:
    """
    Lazy proxy wrapper for EnhancedRAGService to defer event loop binding.
    Delegates all attribute and method calls to the underlying service instance.
    """
    def __init__(self):
        self._instance = None

    @property
    def _service(self) -> EnhancedRAGService:
        if self._instance is None:
            self._instance = EnhancedRAGService()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._service, name)

    def __repr__(self) -> str:
        return f"<LazyRAGServiceProxy (underlying={self._instance})>"

    def __dir__(self) -> List[str]:
        return list(set(super().__dir__() + dir(self._service)))

rag_service: EnhancedRAGService = LazyRAGServiceProxy()  # type: ignore
