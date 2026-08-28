import pytest

def test_rrf_reciprocal_rank_fusion_formula():
    """Verify Reciprocal Rank Fusion (RRF) formula scoring with k=60 constant."""
    # RRF Score = sum(1 / (60 + rank_i))
    k = 60
    # Doc A has rank 1 in lexical (TF-IDF) and rank 2 in dense embedding
    score_doc_a = (1.0 / (k + 1)) + (1.0 / (k + 2))
    
    # Doc B has rank 10 in lexical and rank 1 in dense embedding
    score_doc_b = (1.0 / (k + 10)) + (1.0 / (k + 1))
    
    assert score_doc_a > score_doc_b
    assert score_doc_a == pytest.approx(0.032522, 1e-4)
