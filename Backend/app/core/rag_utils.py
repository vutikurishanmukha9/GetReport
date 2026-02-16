import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import re
import json
import logging

logger = logging.getLogger(__name__)

class TextSplitter:
    """
    Lightweight recursive character text splitter.
    Repluces langchain.text_splitter.RecursiveCharacterTextSplitter
    """
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200, 
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks respecting separators."""
        final_chunks = []
        if not text:
            return []

        # Start with the full text as one chunk
        good_splits = [text]

        for separator in self.separators:
            if not good_splits:
                break
                
            new_splits = []
            for chunk in good_splits:
                if len(chunk) < self.chunk_size:
                    new_splits.append(chunk)
                    continue
                
                # Split this chunk
                splits = chunk.split(separator)
                result_splits = []
                current_chunk = ""
                
                for split in splits:
                    # If adding the next bit exceeds chunk size, save current and start new
                    if len(current_chunk) + len(split) + len(separator) > self.chunk_size:
                        if current_chunk:
                            result_splits.append(current_chunk)
                        current_chunk = split
                    else:
                        current_chunk += (separator if current_chunk else "") + split
                
                if current_chunk:
                    result_splits.append(current_chunk)
                    
                new_splits.extend(result_splits)
            good_splits = new_splits

        return [c.strip() for c in good_splits if c.strip()]

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
            
    def similarity_search_with_score(self, query_embedding: List[float], k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
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

    async def similarity_search_with_score_async(self, query_embedding: List[float], k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """Async search"""
        from app.db import get_async_db_connection
        
        vector_str = f"[{','.join(map(str, query_embedding))}]"
        
        async with get_async_db_connection() as conn:
            # We use our wrapper execute, but we need to handle $n replacement if using ?
            # Wait, here we write raw SQL. Our wrapper expects ?.
            # BUT asyncpg natively uses $n.
            # If we utilize `conn.execute` (our wrapper), we must use `?`.
            # OR we bypass wrapper and use `await conn.conn.fetch` if we want raw power.
            # Let's use our wrapper for consistency of connection management, 
            # but our wrapper expects `?`.
            
            # Query: embedding <=> ?
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

    def similarity_search_with_score(self, query_embedding: List[float], k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """Sync Method"""
        from app.db import get_db_connection
        
        vector_str = f"[{','.join(map(str, query_embedding))}]"
        
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
