import pytest
import polars as pl
from app.services.analysis.core import analyze_dataset

def test_adversarial_unicode_emojis_and_rtl_scripts():
    """Verify processing of international character sets, emojis, RTL Arabic scripts, and Japanese Kanji."""
    df = pl.DataFrame({
        "emoji_labels": ["🚀 Launch", "🔥 Hot", "✅ Done", "⚡ Fast", "💡 Idea"],
        "rtl_arabic": ["مرحبا بالعالم", "بيانات ضخمة", "تحليل إحصائي", "تقرير مالي", "ذكاء اصطناعي"],
        "mixed_japanese": ["売上高", "営業利益", "純利益", "総資産", "自己資本"],
        "values": [100.5, 250.0, 310.2, 490.8, 520.1]
    })
    
    analysis = analyze_dataset(df)
    assert analysis is not None
    assert "summary" in analysis
    assert "values" in analysis["summary"]
    assert analysis["metadata"]["total_rows"] == 5
