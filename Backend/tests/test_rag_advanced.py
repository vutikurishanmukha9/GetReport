import pytest
import json
from app.services.rag_service import (
    _execute_polars_data_query,
    _format_chat_history,
    _generate_suggested_followups,
    _generate_query_variations,
    _extract_targeted_structured_context,
    SecurityGuard
)

def test_polars_data_interpreter_mean_calc():
    job_result = {
        "analysis": {
            "summary": {
                "age": {"mean": 34.5, "min": 18, "max": 75, "std": 12.1, "null_count": 0},
                "income": {"mean": 65000, "min": 20000, "max": 180000}
            },
            "columns": {
                "age": {"type": "numeric"},
                "income": {"type": "numeric"}
            }
        }
    }
    question = "What is the average age of users in the dataset?"
    result = _execute_polars_data_query(question, job_result)
    
    assert "--- VERIFIED EXACT POLARS CALCULATIONS ---" in result
    assert "Column 'age' Exact Verified Stats" in result
    assert "Mean=34.5" in result
    assert "Min=18" in result

def test_polars_data_interpreter_no_math_keywords():
    job_result = {
        "analysis": {
            "summary": {"age": {"mean": 34.5}}
        }
    }
    question = "Where was this data collected?"
    result = _execute_polars_data_query(question, job_result)
    assert result == ""

def test_chat_history_formatter():
    chat_history = [
        {"role": "user", "content": "How many rows are in the dataset?"},
        {"role": "assistant", "content": "There are 10,000 rows in the dataset."},
        {"role": "user", "content": "What is the average age?"}
    ]
    result = _format_chat_history(chat_history)
    assert "--- RECENT CONVERSATION HISTORY ---" in result
    assert "User: How many rows are in the dataset?" in result
    assert "Assistant: There are 10,000 rows in the dataset." in result

def test_security_guard_input_sanitization():
    long_input = "SELECT * FROM users " + ("A" * 3000) + "\x00\x01\x02"
    sanitized = SecurityGuard.sanitize_input(long_input)
    assert len(sanitized) <= 2000
    assert "\x00" not in sanitized
    assert "\x01" not in sanitized

def test_suggested_followups_generation():
    job_result = {
        "analysis": {
            "summary": {
                "age": {"mean": 30},
                "score": {"mean": 85}
            }
        }
    }
    followups = _generate_suggested_followups(job_result)
    assert isinstance(followups, list)
    assert len(followups) == 3
    assert any("age" in f and "score" in f for f in followups)

def test_multi_query_expansion():
    question = "What is the correlation between age and stress_level?"
    variations = _generate_query_variations(question)
    assert len(variations) >= 2
    assert variations[0] == question

def test_targeted_column_context_extraction():
    job_result = {
        "analysis": {
            "summary": {
                "sleep_hours": {"mean": 7.2, "std": 1.1}
            },
            "columns": {
                "sleep_hours": {"type": "numeric"}
            }
        }
    }
    ctx = _extract_targeted_structured_context("Tell me about sleep_hours", job_result)
    assert "--- TARGETED COLUMN STATISTICAL METRICS ---" in ctx
    assert "sleep_hours" in ctx
