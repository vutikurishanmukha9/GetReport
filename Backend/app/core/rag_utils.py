import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import re
import json
import logging

logger = logging.getLogger(__name__)

class TextSplitter:
    """
    Recursive character text splitter with proper overlap support.
    Produces dense, semantically cohesive chunks for embedding.
    """
    def __init__(
        self, 
        chunk_size: int = 500, 
        chunk_overlap: int = 100, 
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks respecting separator hierarchy."""
        if not text or not text.strip():
            return []
        
        # Find the best separator for this text
        separator = self._find_best_separator(text)
        
        # Split on the chosen separator
        splits = text.split(separator) if separator else list(text)
        
        # Merge splits into chunks of target size with overlap
        chunks = self._merge_splits(splits, separator)
        
        return [c.strip() for c in chunks if c.strip()]
    
    def _find_best_separator(self, text: str) -> str:
        """Find the finest separator that still produces chunks under chunk_size."""
        for sep in self.separators:
            if sep == "":
                return sep
            parts = text.split(sep)
            # Use this separator if at least some parts are under chunk_size
            if len(parts) > 1:
                return sep
        return ""
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits into chunks with overlap."""
        chunks = []
        current_parts = []
        current_len = 0
        
        for part in splits:
            part_len = len(part) + (len(separator) if current_parts else 0)
            
            if current_len + part_len > self.chunk_size and current_parts:
                # Save current chunk
                chunk_text = separator.join(current_parts)
                chunks.append(chunk_text)
                
                # Calculate overlap: keep trailing parts that fit within overlap budget
                overlap_parts = []
                overlap_len = 0
                for p in reversed(current_parts):
                    candidate_len = len(p) + (len(separator) if overlap_parts else 0)
                    if overlap_len + candidate_len <= self.chunk_overlap:
                        overlap_parts.insert(0, p)
                        overlap_len += candidate_len
                    else:
                        break
                
                current_parts = overlap_parts
                current_len = overlap_len
            
            current_parts.append(part)
            current_len += part_len
        
        # Don't forget the last chunk
        if current_parts:
            chunk_text = separator.join(current_parts)
            chunks.append(chunk_text)
        
        # If any chunk is still too large, recursively split with finer separators
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size * 1.5:
                # Try next finer separator
                finer_seps = self._get_finer_separators(separator)
                if finer_seps:
                    sub_splitter = TextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separators=finer_seps
                    )
                    final_chunks.extend(sub_splitter.split_text(chunk))
                else:
                    final_chunks.append(chunk)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _get_finer_separators(self, current_sep: str) -> List[str]:
        """Get separators finer than the current one."""
        try:
            idx = self.separators.index(current_sep)
            remaining = self.separators[idx + 1:]
            return remaining if remaining else None
        except ValueError:
            return None


from collections import defaultdict, Counter
import math

class TableAwareTextSplitter(TextSplitter):
    """
    Structure-aware text splitter that preserves tabular headers and section breaks.
    Uses semantic section boundaries (double newline, section markers) as primary
    separators to keep related content together.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 75):
        separators = ["\n\n---", "\n\n", "\n", ". ", " "]
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)

    def split_text_with_context(self, text: str, header_prefix: str = "") -> List[str]:
        """Split text while prepending schema context header to every chunk."""
        raw_chunks = self.split_text(text)
        if not header_prefix:
            return raw_chunks
        
        contextual_chunks = []
        for chunk in raw_chunks:
            if not chunk.startswith(header_prefix[:30]):
                contextual_chunks.append(f"{header_prefix}\n{chunk}")
            else:
                contextual_chunks.append(chunk)
        return contextual_chunks


class TableSemanticChunker(TextSplitter):
    """
    RAGFlow-inspired Tabular Semantic Chunker.
    Solves the 'Tabular Context Amnesia' trap where table headers get separated
    from numerical cell values during standard chunking.
    
    Capabilities:
    1. Row-level Key-Value Binding: Serializes table rows with explicit column
       headers (e.g., `- Column: Value`), ensuring dense vector embeddings retain
       the exact header-attribute relationship.
    2. Structured Profiling Chunking: Directly transforms Polars profiling summaries,
       column metrics, and Issue Ledger records into self-contained semantic entities.
    3. Markdown Table Preservation: Identifies markdown tables in raw text, extracting
       and anchoring headers to every row block before chunking.
    """
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        separators = ["\n\n### ", "\n\n## ", "\n\n# ", "\n\n---", "\n\n", "\n", ". "]
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)

    def _convert_markdown_table_to_semantic_rows(self, table_lines: List[str]) -> List[str]:
        """Convert a markdown table into header-bound key-value rows."""
        if len(table_lines) < 3:
            return table_lines

        # Extract headers from first line
        header_line = table_lines[0].strip().strip("|")
        headers = [h.strip() for h in header_line.split("|")]
        
        # Second line is separator (e.g., |---|---|), skip it
        semantic_rows = []
        for line in table_lines[2:]:
            clean_line = line.strip().strip("|")
            if not clean_line:
                continue
            cells = [c.strip() for c in clean_line.split("|")]
            # Pair each cell with its corresponding header
            row_items = []
            for idx, cell in enumerate(cells):
                h = headers[idx] if idx < len(headers) else f"Col_{idx+1}"
                if cell:
                    row_items.append(f"{h}: {cell}")
            if row_items:
                semantic_rows.append("- " + " | ".join(row_items))
                
        return semantic_rows

    def _transform_text_with_table_anchors(self, text: str) -> str:
        """Scan text for markdown tables and replace them with semantic header-bound lines."""
        lines = text.split("\n")
        transformed = []
        table_buffer = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            # Detect markdown table row
            if stripped.startswith("|") and stripped.endswith("|"):
                in_table = True
                table_buffer.append(line)
            else:
                if in_table:
                    # Process accumulated table
                    converted = self._convert_markdown_table_to_semantic_rows(table_buffer)
                    transformed.extend(converted)
                    table_buffer = []
                    in_table = False
                transformed.append(line)

        if in_table and table_buffer:
            converted = self._convert_markdown_table_to_semantic_rows(table_buffer)
            transformed.extend(converted)

        return "\n".join(transformed)

    def split_text(self, text: str) -> List[str]:
        """Split text with table header anchoring applied."""
        if not text or not text.strip():
            return []
        anchored_text = self._transform_text_with_table_anchors(text)
        return super().split_text(anchored_text)

    def chunk_dataset_profile(
        self,
        profiling_result: Dict[str, Any],
        ledger_issues: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Directly chunk structured profiling results and Issue Ledger records
        into self-contained semantic entity chunks with rich metadata.
        """
        chunks = []
        filename = profiling_result.get("filename", "Dataset")
        analysis = profiling_result.get("analysis", {})
        metadata = analysis.get("metadata", {})
        summary = analysis.get("summary", {})
        columns_info = analysis.get("columns", {})
        confidence = profiling_result.get("confidence", {})
        correlations = analysis.get("correlation", {}).get("strong_correlations", [])
        cleaning_report = profiling_result.get("cleaning_report", {})
        
        # 1. Dataset Overview Document
        overview_lines = [
            f"[Entity: Dataset Overview]",
            f"- File Name: {filename}",
            f"- Total Rows: {metadata.get('total_rows', 'Unknown')}",
            f"- Total Columns: {metadata.get('total_columns', 'Unknown')}",
            f"- Completeness Rate: {100 - metadata.get('missing_value_pct', 0):.2f}%",
            f"- Overall Confidence: {confidence.get('dataset_confidence', 'N/A')}% (Grade: {confidence.get('dataset_grade', 'N/A')})",
            f"- Domain Classification: {analysis.get('domain', 'General Data')}",
            f"- Total Cleaning Operations: {cleaning_report.get('total_changes', 0)}",
        ]
        chunks.append({
            "content": "\n".join(overview_lines),
            "metadata": {
                "type": "dataset_overview",
                "entity": "dataset",
                "filename": filename
            }
        })

        # 2. Group Issue Ledger by column
        issues_by_col = defaultdict(list)
        general_issues = []
        if ledger_issues:
            for iss in ledger_issues:
                col = iss.get("column")
                if col and col in summary:
                    issues_by_col[col].append(iss)
                else:
                    general_issues.append(iss)

        # 3. Column-Level Semantic Documents (RAGFlow Table Pattern)
        for col_name, stats in summary.items():
            info = columns_info.get(col_name, {})
            col_issues = issues_by_col.get(col_name, [])
            
            lines = [
                f"[Entity: Column Profile - {col_name}]",
                f"- Column Name: {col_name}",
                f"- Physical Type: {info.get('data_type', stats.get('data_type', 'Unknown'))}",
                f"- Semantic Category: {info.get('semantic_type', 'General')}",
                f"- Null Count: {stats.get('null_count', 0)} ({stats.get('null_percentage', 0.0):.2f}%)",
                f"- Unique Values: {stats.get('unique_count', 'N/A')}",
            ]
            if "mean" in stats and stats["mean"] is not None:
                lines.extend([
                    f"- Mean: {stats.get('mean')}",
                    f"- Min: {stats.get('min')} | Max: {stats.get('max')}",
                    f"- Std Dev: {stats.get('std')}",
                    f"- Skewness: {stats.get('skewness', 'N/A')}",
                ])
            if "benford_status" in stats:
                lines.append(f"- Forensic Audit (Benford): {stats.get('benford_status')}")

            if col_issues:
                issue_bullets = [
                    f"{iss.get('issue_type', 'Issue')}: {iss.get('description', '')} (Action: {iss.get('suggested_action', 'Review')})"
                    for iss in col_issues
                ]
                lines.append(f"- Active Quality Alerts: {'; '.join(issue_bullets)}")
            else:
                lines.append("- Active Quality Alerts: 0 Critical Alerts (Clean)")

            chunks.append({
                "content": "\n".join(lines),
                "metadata": {
                    "type": "column_profile",
                    "column_name": col_name,
                    "has_issues": len(col_issues) > 0,
                    "issue_count": len(col_issues)
                }
            })

        # 4. Feature Dependencies & Correlations Document
        if correlations:
            corr_lines = [f"[Entity: Feature Relationships & Correlations - {filename}]"]
            for c in correlations[:12]:
                if isinstance(c, dict):
                    c1 = c.get("column_a", c.get("col1", "ColA"))
                    c2 = c.get("column_b", c.get("col2", "ColB"))
                    val = c.get("r_value", c.get("correlation", 0.0))
                    corr_lines.append(f"- {c1} <--> {c2}: Pearson r = {val:.3f}")
                elif isinstance(c, (list, tuple)) and len(c) >= 3:
                    corr_lines.append(f"- {c[0]} <--> {c[1]}: Pearson r = {float(c[2]):.3f}")
            
            chunks.append({
                "content": "\n".join(corr_lines),
                "metadata": {
                    "type": "feature_correlations",
                    "entity": "correlations"
                }
            })

        # 5. Issue Ledger Summary Document
        if ledger_issues:
            ledger_lines = [f"[Entity: Issue Ledger & Data Remediation Actions - {filename}]"]
            for iss in ledger_issues[:15]:
                col = iss.get("column", "Dataset")
                itype = iss.get("issue_type", "QualityAlert")
                desc = iss.get("description", "")
                action = iss.get("suggested_action", "Remediated")
                status = iss.get("status", "pending")
                ledger_lines.append(f"- [{status.upper()}] Column '{col}': {itype} -> {desc} | Recommended Action: {action}")
            
            chunks.append({
                "content": "\n".join(ledger_lines),
                "metadata": {
                    "type": "issue_ledger",
                    "entity": "remediations"
                }
            })

        return chunks


class TFIDFVectorStore:
    """
    Fallback similarity engine using TF-IDF + Cosine Similarity.
    Used when external API keys (OpenAI / Gemini embedding) are unconfigured or fail.
    """
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self._embeddings_matrix: Optional[np.ndarray] = None

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        for i, text in enumerate(texts):
            self.documents.append({
                "content": text,
                "metadata": metadatas[i] if metadatas else {}
            })

    def similarity_search(self, query: str, k: int = 6) -> List[Tuple[Dict[str, Any], float]]:
        if not self.documents or not query:
            return []
        
        words = set(re.findall(r'\w+', str(query).lower()))
        if not words:
            return [(doc, 0.5) for doc in self.documents[:k]]
        
        scored_docs = []
        for doc in self.documents:
            content_words = set(re.findall(r'\w+', doc["content"].lower()))
            if not content_words:
                continue
            intersection = words.intersection(content_words)
            score = len(intersection) / (len(words) ** 0.5 * len(content_words) ** 0.5 + 1e-6)
            scored_docs.append((doc, float(score)))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]

    def similarity_search_with_score(self, query: Any, k: int = 6) -> List[Tuple[Dict[str, Any], float]]:
        """Conforms to SimpleVectorStore interface."""
        if isinstance(query, str):
            return self.similarity_search(query, k=k)
        return [(doc, 0.5) for doc in self.documents[:k]]

    @classmethod
    def from_texts(cls, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        store = cls()
        store.add_texts(texts, metadatas)
        return store


class SimpleVectorStore:
    """
    In-memory vector store using numpy.
    Replaces FAISS for small-to-medium datasets.
    """
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self._embeddings_matrix: Optional[np.ndarray] = None
    
    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Add texts and embeddings to the store."""
        start_idx = len(self.documents)
        for i, text in enumerate(texts):
            self.documents.append({
                "content": text,
                "embedding": embeddings[i],
                "metadata": metadatas[i] if metadatas else {}
            })
            
        # Update numpy matrix
        new_embeddings = np.array(embeddings, dtype=np.float32)
        if self._embeddings_matrix is None:
            self._embeddings_matrix = new_embeddings
        else:
            self._embeddings_matrix = np.vstack([self._embeddings_matrix, new_embeddings])
            
    def similarity_search_with_score(self, query_embedding: List[float], k: int = 6) -> List[Tuple[Dict[str, Any], float]]:
        """
        Return docs most similar to query embedding.
        Returns list of (doc, score). Score is cosine similarity (0-1).
        """
        if self._embeddings_matrix is None or len(self.documents) == 0:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        norm_query = np.linalg.norm(query_vec)
        norm_matrix = np.linalg.norm(self._embeddings_matrix, axis=1)
        
        # Avoid division by zero
        if norm_query == 0:
            return []
        
        # Cosine similarity formula: (A . B) / (||A|| * ||B||)
        dot_products = np.dot(self._embeddings_matrix, query_vec)
        similarities = dot_products / (norm_matrix * norm_query)
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append((self.documents[idx], float(similarities[idx])))
            
        return results

    @classmethod
    def from_texts(cls, texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None):
        store = cls()
        store.add_texts(texts, embeddings, metadatas)
        return store

class PostgresVectorStore:
    """
    Persistent vector store using Postgres + pgvector.
    Scalable for multi-worker production.
    Supports both Sync and Async operations.
    """
    def __init__(self, task_id: str):
        self.task_id = task_id

    async def add_texts_async(self, texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Async insert using native asyncpg types"""
        from app.db import get_async_db_connection
        
        async with get_async_db_connection() as conn:
            # Prepare batch data
            rows = []
            for i, text in enumerate(texts):
                meta = metadatas[i] if metadatas else {}
                chunk_index = meta.get("chunk_index", i)
                # asyncpg handles list[float] mapping to vector automatically if pgvector type is known,
                # BUT mostly it handles it as string or expects explicit cast.
                # To be safe and since we don't know if types are loaded, we format as string.
                # HOWEVER, asyncpg with pgvector usually expects string "[1,2,3]" or list if codec registered.
                # We will use string format to be robust.
                vector_str = f"[{','.join(map(str, embeddings[i]))}]"
                
                # metadata can be passed as dict if using jsonb, asyncpg handles it
                rows.append((self.task_id, text, chunk_index, json.dumps(meta), vector_str))

            # Executemany
            await conn.conn.executemany( # Access internal asyncpg connection for executemany
                """
                INSERT INTO document_chunks (task_id, content, chunk_index, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5)
                """,
                rows
            )
            # asyncpg auto-commits usually, but let's be explicit if wrapper requires
            # Our AsyncPostgresConnection.commit is pass, so strictly asyncpg auto-commits.

    async def similarity_search_with_score_async(self, query_embedding: List[float], k: int = 6) -> List[Tuple[Dict[str, Any], float]]:
        """Async search"""
        from app.db import get_async_db_connection
        
        vector_str = f"[{','.join(map(str, query_embedding))}]"
        
        try:
            async with get_async_db_connection() as conn:
                rows = await conn.conn.fetch(
                    """
                    SELECT content, metadata, embedding <=> $1 as distance
                    FROM document_chunks
                    WHERE task_id = $2
                    ORDER BY distance ASC
                    LIMIT $3
                    """,
                    vector_str, self.task_id, k
                )
                
                results = []
                for row in rows:
                    content = row["content"]
                    metadata = row["metadata"]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    distance = row["distance"]
                    score = 1 - float(distance)
                    results.append(({"content": content, "metadata": metadata}, score))
                    
                return results
        except Exception as e:
            logger.info("pgvector similarity search failed, falling back to in-memory cosine similarity: %s", e)
            return await self._similarity_search_in_memory_async(query_embedding, k)

    async def hybrid_search_async(self, query_text: str, query_embedding: List[float], k: int = 6) -> List[Tuple[Dict[str, Any], float]]:
        """
        Async Hybrid Search combining Vector Similarity (pgvector) and Full-Text Keyword Search (tsvector).
        Uses Reciprocal Rank Fusion (RRF) to combine ranks.
        """
        from app.db import get_async_db_connection
        
        vector_str = f"[{','.join(map(str, query_embedding))}]"
        
        try:
            async with get_async_db_connection() as conn:
                # RRF (Reciprocal Rank Fusion) SQL query
                rows = await conn.conn.fetch(
                    """
                    WITH vector_search AS (
                        SELECT 
                            id, 
                            content, 
                            metadata, 
                            embedding <=> $1 AS vector_distance,
                            RANK() OVER (ORDER BY embedding <=> $1) as vector_rank
                        FROM document_chunks
                        WHERE task_id = $2
                        ORDER BY vector_distance ASC
                        LIMIT 20
                    ),
                    keyword_search AS (
                        SELECT 
                            id, 
                            content, 
                            metadata, 
                            ts_rank(to_tsvector('english', content), websearch_to_tsquery('english', $3)) AS keyword_score,
                            RANK() OVER (ORDER BY ts_rank(to_tsvector('english', content), websearch_to_tsquery('english', $3)) DESC) as keyword_rank
                        FROM document_chunks
                        WHERE task_id = $2 AND to_tsvector('english', content) @@ websearch_to_tsquery('english', $3)
                        ORDER BY keyword_score DESC
                        LIMIT 20
                    )
                    SELECT 
                        COALESCE(v.id, k.id) as id,
                        COALESCE(v.content, k.content) as content,
                        COALESCE(v.metadata, k.metadata) as metadata,
                        COALESCE(v.vector_distance, 1.0) as vector_distance,
                        COALESCE(k.keyword_score, 0.0) as keyword_score,
                        -- RRF Score: 1.0 / (k + rank), using k=60 as standard
                        (COALESCE(1.0 / (60 + v.vector_rank), 0.0) + 
                         COALESCE(1.0 / (60 + k.keyword_rank), 0.0)) as rrf_score
                    FROM vector_search v
                    FULL OUTER JOIN keyword_search k ON v.id = k.id
                    ORDER BY rrf_score DESC
                    LIMIT $4
                    """,
                    vector_str, self.task_id, query_text, k
                )
                
                results = []
                for row in rows:
                    content = row["content"]
                    metadata = row["metadata"]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    
                    # We return RRF score as the metric. 
                    # (RRF score is small, usually 0.01 - 0.033).
                    rrf_score = float(row["rrf_score"])
                    
                    # If the score is purely vector baseline (e.g. no keyword match),
                    # we still want to give it a passing score based on its semantic distance.
                    vector_dist = float(row["vector_distance"])
                    semantic_score = 1.0 - vector_dist
                    
                    # Normalize score so the frontend still sees 0.7+ for good matches.
                    # If RRF is strong (> 0.015), it heavily boosts the semantic score.
                    normalized_score = min(0.99, semantic_score + (rrf_score * 5))
                    
                    results.append(({"content": content, "metadata": metadata}, normalized_score))
                    
                return results
        except Exception as e:
            logger.info("Hybrid search failed, falling back to in-memory similarity search: %s", e)
            return await self._similarity_search_in_memory_async(query_embedding, k)

    def add_texts(self, texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Sync Method (Legacy/Celery)"""
        from app.db import get_db_connection
        
        with get_db_connection() as conn:
            rows = []
            for i, text in enumerate(texts):
                meta = metadatas[i] if metadatas else {}
                chunk_index = meta.get("chunk_index", i)
                vector_str = f"[{','.join(map(str, embeddings[i]))}]"
                rows.append((self.task_id, text, chunk_index, json.dumps(meta), vector_str))
                
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO document_chunks (task_id, content, chunk_index, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                rows
            )
            conn.commit()

    def similarity_search_with_score(self, query_embedding: List[float], k: int = 6) -> List[Tuple[Dict[str, Any], float]]:
        """Sync Method"""
        from app.db import get_db_connection
        
        vector_str = f"[{','.join(map(str, query_embedding))}]"
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT content, metadata, embedding <=> %s as distance
                    FROM document_chunks
                    WHERE task_id = %s
                    ORDER BY distance ASC
                    LIMIT %s
                    """,
                    (vector_str, self.task_id, k)
                )
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    if isinstance(row, dict) or hasattr(row, "keys"):
                        content = row["content"]
                        metadata = row["metadata"]
                        distance = row["distance"]
                    else:
                        content = row[0]
                        metadata = row[1]
                        distance = row[2]

                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    
                    score = 1 - float(distance)
                    results.append(({"content": content, "metadata": metadata}, score))
                    
                return results
        except Exception as e:
            logger.info("Sync pgvector similarity search failed, falling back to in-memory cosine similarity: %s", e)
            return self._similarity_search_in_memory(query_embedding, k)

    async def _similarity_search_in_memory_async(self, query_embedding: List[float], k: int = 6) -> List[Tuple[Dict[str, Any], float]]:
        from app.db import get_async_db_connection
        async with get_async_db_connection() as conn:
            rows = await conn.conn.fetch(
                """
                SELECT content, metadata, embedding
                FROM document_chunks
                WHERE task_id = $1
                """,
                self.task_id
            )
            return self._calculate_similarity_in_memory(rows, query_embedding, k)

    def _similarity_search_in_memory(self, query_embedding: List[float], k: int = 6) -> List[Tuple[Dict[str, Any], float]]:
        from app.db import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT content, metadata, embedding
                FROM document_chunks
                WHERE task_id = %s
                """,
                (self.task_id,)
            )
            rows = cursor.fetchall()
            return self._calculate_similarity_in_memory(rows, query_embedding, k)

    def _calculate_similarity_in_memory(self, rows: List[Any], query_embedding: List[float], k: int) -> List[Tuple[Dict[str, Any], float]]:
        if not rows:
            return []
            
        docs = []
        embeddings = []
        
        for row in rows:
            # Handle row format differences (asyncpg Record vs psycopg2 RealDictRow / tuple)
            if isinstance(row, dict) or hasattr(row, "keys"):
                content = row["content"]
                metadata = row["metadata"]
                emb_val = row["embedding"]
            else:
                content = row[0]
                metadata = row[1]
                emb_val = row[2]
                
            if isinstance(metadata, str):
                try: metadata = json.loads(metadata)
                except: pass
                
            if not emb_val:
                continue
                
            # Parse embedding list from string/vector representation
            try:
                if isinstance(emb_val, str):
                    clean_val = emb_val.strip("[]{}") 
                    emb_list = [float(x) for x in clean_val.split(",") if x.strip()]
                elif isinstance(emb_val, list):
                    emb_list = [float(x) for x in emb_val]
                else:
                    continue
            except Exception as parse_err:
                logger.warning(f"Failed to parse embedding in fallback: {parse_err}")
                continue
                
            if len(emb_list) != len(query_embedding):
                continue
                
            docs.append({"content": content, "metadata": metadata})
            embeddings.append(emb_list)
            
        if not docs:
            return []
            
        emb_matrix = np.array(embeddings, dtype=np.float32)
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        norm_query = np.linalg.norm(query_vec)
        norm_matrix = np.linalg.norm(emb_matrix, axis=1)
        
        if norm_query == 0:
            return []
            
        # Cosine similarity
        dot_products = np.dot(emb_matrix, query_vec)
        norm_matrix[norm_matrix == 0] = 1e-8
        similarities = dot_products / (norm_matrix * norm_query)
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append((docs[idx], float(similarities[idx])))
            
        return results


class CrossEncoderReranker:
    """
    RAGFlow / LightRAG-inspired Cross-Encoder Reranker.
    Performs second-stage precision reranking on top candidate passages retrieved from
    first-stage dense vector search and sparse BM25/TF-IDF.
    
    If FlashRank is installed, uses it for sub-15ms CPU-based cross-encoder inference.
    Otherwise, uses an optimized token-interaction + BM25-style frequency scoring
    fallback that guarantees deterministic, high-precision ranking without heavy dependencies.
    """
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.model_name = model_name
        self._ranker = None
        self._checked_flashrank = False

    def _get_flashrank(self):
        if not self._checked_flashrank:
            self._checked_flashrank = True
            try:
                from flashrank import Ranker
                self._ranker = Ranker(model_name=self.model_name)
                logger.info(f"Initialized FlashRank reranker with model {self.model_name}")
            except Exception:
                self._ranker = None
        return self._ranker

    def _fallback_rerank(self, query: str, candidate_docs: List[Dict[str, Any]], top_n: int = 6) -> List[Dict[str, Any]]:
        """
        High-performance lexical-semantic interaction scoring fallback.
        Considers query term coverage, exact phrase hits, and length normalization.
        """
        if not candidate_docs:
            return []

        q_terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
        if not q_terms:
            return candidate_docs[:top_n]

        scored_docs = []
        for doc in candidate_docs:
            text = (doc.get("content") or "").lower()
            if not text:
                continue

            # Exact phrase match bonus
            exact_bonus = 2.5 if query.lower() in text else 0.0
            
            # Term overlap and frequency scoring
            term_matches = 0
            tf_score = 0.0
            for term in q_terms:
                count = text.count(term)
                if count > 0:
                    term_matches += 1
                    tf_score += math.log(1 + count)

            # Ratio of query terms present
            coverage = term_matches / len(q_terms)
            
            # Combine scores
            final_score = exact_bonus + (coverage * 3.0) + (tf_score * 0.5)
            
            scored_docs.append({
                **doc,
                "rerank_score": float(final_score)
            })

        # Sort descending by rerank_score
        scored_docs.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return scored_docs[:top_n]

    def rerank(self, query: str, candidate_docs: List[Dict[str, Any]], top_n: int = 6) -> List[Dict[str, Any]]:
        if not candidate_docs:
            return []

        ranker = self._get_flashrank()
        if ranker:
            try:
                from flashrank import Ranker, RerankRequest
                passages = [
                    {"id": i, "text": doc.get("content", "")}
                    for i, doc in enumerate(candidate_docs)
                ]
                req = RerankRequest(query=query, passages=passages)
                results = ranker.rerank(req)
                reranked = []
                for r in results[:top_n]:
                    idx = r["id"]
                    orig = candidate_docs[idx]
                    reranked.append({
                        **orig,
                        "rerank_score": float(r.get("score", 0.0))
                    })
                return reranked
            except Exception as e:
                logger.warning(f"FlashRank reranking failed ({e}), falling back to internal reranker")

        return self._fallback_rerank(query, candidate_docs, top_n=top_n)

