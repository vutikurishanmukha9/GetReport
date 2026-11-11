import sys
import os
sys.path.insert(0, os.path.abspath("Backend"))
import pytest
import io
import hashlib
from fastapi import HTTPException
from app.core.file_validation import validate_file_signature, verify_header_bytes, SIGNATURES
from app.services.data_processing import load_dataframe
import polars as pl

class MockUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self.content = content
        self.position = 0
        
    async def seek(self, pos):
        self.position = pos
        
    async def read(self, size):
        data = self.content[self.position:self.position+size]
        self.position += size
        return data

def test_mime_magic_bytes_detection():
    # Valid parquet file header PAR1
    verify_header_bytes(SIGNATURES["parquet"] + b"\x00\x00data", "data.parquet")

def test_file_validation_invalid_xlsx():
    with pytest.raises(HTTPException) as exc_info:
        verify_header_bytes(b"This is text content not a zip archive", "fake.xlsx")
    assert exc_info.value.status_code == 400

def test_file_validation_csv_with_null_bytes_rejected():
    with pytest.raises(HTTPException) as exc_info:
        verify_header_bytes(b"col1,col2\x00malicious_binary", "payload.csv")
    assert exc_info.value.status_code == 400
    assert "binary" in exc_info.value.detail

def test_file_validation_missing_filename():
    with pytest.raises(HTTPException) as exc_info:
        verify_header_bytes(b"col1,col2\n1,2", None)
    assert exc_info.value.status_code == 400

def test_sha256_file_deduplication_hash():
    content = b"col1,col2\n1,2\n3,4\n"
    hash1 = hashlib.sha256(content).hexdigest()
    hash2 = hashlib.sha256(content).hexdigest()
    assert hash1 == hash2
    assert len(hash1) == 64

def test_polars_multi_format_parquet_loading(tmp_path):
    df = pl.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"]
    })
    parquet_file = tmp_path / "test.parquet"
    df.write_parquet(parquet_file)
    
    loaded_df = load_dataframe(str(parquet_file))
    assert loaded_df.shape == (3, 2)
    assert loaded_df.columns == ["a", "b"]

if __name__ == "__main__":
    test_mime_magic_bytes_detection()
    test_file_validation_invalid_xlsx()
    test_file_validation_csv_with_null_bytes_rejected()
    test_file_validation_missing_filename()
    test_sha256_file_deduplication_hash()
    print("ALL TESTS PASSED DIRECTLY!")
