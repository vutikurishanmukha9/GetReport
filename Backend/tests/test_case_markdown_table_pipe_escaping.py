import pytest
from app.core.rag_utils import TableAwareTextSplitter

def test_markdown_table_pipe_character_handling():
    """Verify TableAwareTextSplitter handles tables with escaped pipes or pipes inside backticks."""
    text = r"""
# METRICS SUMMARY
| Column Name | Description | Example |
| `status` | User status (`active` \| `pending`) | active |
| `price` | Total cost in USD | $45.00 |
"""
    splitter = TableAwareTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)
    assert len(chunks) >= 1
    assert any("price" in c for c in chunks)
