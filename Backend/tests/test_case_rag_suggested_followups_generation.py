import pytest
from app.services.rag_service import _generate_suggested_followups

def test_rag_suggested_followups_generation():
    """Verify contextual suggested follow-up questions are generated based on dataset columns."""
    job_result = {
        "analysis": {
            "summary": {
                "sales": {"mean": 100.0},
                "ad_spend": {"mean": 50.0}
            }
        }
    }
    
    followups = _generate_suggested_followups(job_result)
    assert len(followups) >= 1
    assert any("sales" in f.lower() or "correlation" in f.lower() or "quality" in f.lower() for f in followups)
