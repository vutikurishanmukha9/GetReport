import pytest
import polars as pl
from app.services.foreign_key_integrity import detect_foreign_key_violations

def test_foreign_key_referential_completeness_100_percent():
    """Verify that a perfect child table with zero missing foreign keys reports 100% integrity."""
    parent_df = pl.DataFrame({"category_id": [1, 2, 3, 4, 5]})
    child_df = pl.DataFrame({
        "product_id": [101, 102, 103, 104],
        "category_id": [1, 3, 2, 5]
    })
    
    res = detect_foreign_key_violations(
        primary_df=parent_df,
        foreign_df=child_df,
        pk_col="category_id",
        fk_col="category_id"
    )
    assert res["has_issue"] is False
    assert res["orphan_count"] == 0
    assert res["referential_completeness"] == 100.0
