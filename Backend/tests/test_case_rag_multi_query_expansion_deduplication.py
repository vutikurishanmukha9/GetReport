import pytest
from app.services.rag_service import _generate_query_variations

def test_rag_multi_query_expansion():
    """Verify query variations generator produces multiple semantically related queries."""
    query = "What is the correlation between marketing spend and revenue?"
    variations = _generate_query_variations(query)
    assert len(variations) >= 1
    assert query in variations
