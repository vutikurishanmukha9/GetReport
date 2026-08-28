import pytest
import polars as pl
from app.services.issue_ledger import compute_dataset_fingerprint

def test_dataset_semantic_fingerprint_generation():
    """Verify that dataset semantic fingerprinting generates deterministic 64-char SHA-256 hashes."""
    df1 = pl.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.0, 30.2]})
    df2 = pl.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.0, 30.2]})
    df3 = pl.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.0, 999.9]})
    
    fp1 = compute_dataset_fingerprint(df1)
    fp2 = compute_dataset_fingerprint(df2)
    fp3 = compute_dataset_fingerprint(df3)
    
    assert fp1 == fp2
    assert fp1 != fp3
    assert len(fp1) == 64
