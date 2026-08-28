import pytest
import polars as pl
from app.services.feature_engineering import _suggest_text_features

def test_feature_engineering_suggests_text_length_and_tokens():
    """Verify text feature extractor suggests character length and word count features."""
    df = pl.DataFrame({
        "feedback": [
            "Great fast shipping and wonderful product!",
            "Broken item upon arrival. Very disappointed.",
            "Customer support resolved it promptly.",
            "Five stars!"
        ]
    })
    
    sugg = _suggest_text_features(df, "feedback")
    assert sugg.category == "text"
    assert len(sugg.suggested_features) >= 2
    names = [f["name"] for f in sugg.suggested_features]
    assert "feedback_length" in names
    assert "feedback_word_count" in names
