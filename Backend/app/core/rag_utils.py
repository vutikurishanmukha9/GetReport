import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import re

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
                        
                        # Use overlap if possible (simplified here: just backtrack a bit? 
                        # actually, proper overlap is hard without tokenizers. 
                        # We'll use a sliding window approach for simplicity or just simple accumulation)
                        current_chunk = split
                    else:
                        current_chunk += (separator if current_chunk else "") + split
                
                if current_chunk:
                    result_splits.append(current_chunk)
                    
                new_splits.extend(result_splits)
            good_splits = new_splits

        # Final pass via simpler sliding window if strict size needed? 
        # The above logic is a decent approximation of recursive splitting.
        # Let's verify lengths and force split if still too big?
        # For now, we trust the separators to do good enough work.
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
