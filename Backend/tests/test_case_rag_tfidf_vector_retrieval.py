import pytest
from app.core.rag_utils import TFIDFVectorStore

def test_rag_tfidf_vector_store_retrieval():
    """Verify TF-IDF vector store correctly retrieves top matching text chunks based on terms."""
    store = TFIDFVectorStore()
    
    docs = [
        "The quick brown fox jumps over the lazy dog and runs in the park.",
        "Revenue increased by 25% year-over-year driven by cloud subscription sales.",
        "Data quality audit found 15 missing values in the customer age column.",
        "Machine learning readiness score is evaluated at 92.5% indicating strong signal."
    ]
    metadatas = [{"source": f"doc_{i}"} for i in range(len(docs))]
    
    store.add_texts(docs, metadatas=metadatas)
    
    results = store.similarity_search("What happened to subscription revenue?", k=2)
    assert len(results) >= 1
    assert "Revenue" in results[0][0]["content"]
    
    results_missing = store.similarity_search("Are there any null or missing values in age?", k=2)
    assert len(results_missing) >= 1
    assert "missing values" in results_missing[0][0]["content"]
