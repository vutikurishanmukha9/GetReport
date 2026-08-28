import pytest
import polars as pl
from app.services.foreign_key_integrity import detect_foreign_key_violations

def test_foreign_key_orphan_records_detection():
    """Verify detection of orphaned foreign key references between child and parent tables."""
    parent_df = pl.DataFrame({
        "user_id": [1, 2, 3, 4],
        "username": ["Alice", "Bob", "Charlie", "Diana"]
    })
    
    child_df = pl.DataFrame({
        "order_id": [101, 102, 103, 104, 105],
        "user_id": [1, 2, 999, 4, 888] # 999 and 888 are orphan keys
    })
    
    integrity_res = detect_foreign_key_violations(
        primary_df=parent_df,
        foreign_df=child_df,
        pk_col="user_id",
        fk_col="user_id"
    )
    assert integrity_res["has_issue"] is True
    assert integrity_res["orphan_count"] == 2
    assert 999 in integrity_res["sample_orphan_keys"] or 888 in integrity_res["sample_orphan_keys"]
