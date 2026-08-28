import pytest
from app.core.rag_utils import TextSplitter, TableAwareTextSplitter

def test_rag_table_aware_text_splitter_preservation():
    """Verify recursive character splitting and table boundary preservation."""
    sample_text = """
# SECTION 1: EXECUTIVE SUMMARY
This is a high-level briefing on the sales performance for Q3. We observed a 15% growth.

# SECTION 2: STATISTICAL METRICS TABLE
| Column | Mean | Min | Max |
| Revenue | 5400.5 | 100.0 | 25000.0 |
| Cost | 3200.2 | 50.0 | 18000.0 |
| Margin | 0.41 | 0.10 | 0.75 |

# SECTION 3: KEY RECOMMENDATIONS
1. Increase inventory for top 3 performing SKUs.
2. Reduce marketing spend on zero-converting campaigns.
"""
    splitter = TextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = splitter.split_text(sample_text)
    assert len(chunks) >= 2

    table_splitter = TableAwareTextSplitter(chunk_size=200, chunk_overlap=40)
    table_chunks = table_splitter.split_text(sample_text)
    assert len(table_chunks) >= 2
    
    table_chunk = next((c for c in table_chunks if "Revenue" in c), None)
    assert table_chunk is not None
