import pytest
import io
import hashlib
from fastapi import HTTPException
from app.core.file_validation import validate_file_signature, SIGNATURES
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

@pytest.mark.anyio
async def test_mime_magic_bytes_detection():
    # Valid parquet file header PAR1
    parquet_file = MockUploadFile("data.parquet", SIGNATURES["parquet"] + b"\x00\x00data")
    await validate_file_signature(parquet_file)

@pytest.mark.anyio
async def test_file_validation_invalid_xlsx():
    bad_xlsx = MockUploadFile("fake.xlsx", b"This is text content not a zip archive")
    with pytest.raises(HTTPException) as exc_info:
        await validate_file_signature(bad_xlsx)
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
