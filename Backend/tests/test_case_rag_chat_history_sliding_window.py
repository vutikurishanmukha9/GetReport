import pytest
from app.services.rag_service import _format_chat_history

def test_rag_chat_history_formatting():
    """Verify chat history correctly formats user and assistant messages for LLM context."""
    history = [
        {"role": "user", "content": "How many missing values are there?"},
        {"role": "assistant", "content": "There are 15 missing values in age."},
        {"role": "user", "content": "What about salary?"}
    ]
    
    formatted = _format_chat_history(history)
    assert "User: How many missing values" in formatted
    assert "Assistant: There are 15 missing" in formatted
    assert "User: What about salary?" in formatted
