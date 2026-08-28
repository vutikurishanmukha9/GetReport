import pytest
from app.core.rag_utils import SimpleVectorStore

def test_rag_simple_vector_store_cosine_matching():
    """Verify dense embedding vector storage and cosine similarity scoring."""
    store = SimpleVectorStore()
    
    texts = ["Machine Learning model training", "Accounting balance sheet report"]
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0]
    ]
    
    store.add_texts(texts, embeddings)
    
    query_vector = [0.95, 0.05]
    matches = store.similarity_search_with_score(query_vector, k=2)
    assert len(matches) == 2
    assert "Machine Learning" in matches[0][0]["content"]
    assert matches[0][1] > 0.90
