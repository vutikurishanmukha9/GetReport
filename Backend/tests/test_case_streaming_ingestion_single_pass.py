import io
import os
import hashlib
import asyncio
import pytest
from starlette.datastructures import UploadFile
from fastapi import HTTPException
from app.services.storage import LocalStorageProvider
from app.core.file_validation import verify_header_bytes, SIGNATURES

@pytest.mark.anyio
async def test_single_pass_streaming_valid_csv(tmp_path):
    """Verify single-pass streaming accurately hashes, sizes, and saves CSV data."""
    provider = LocalStorageProvider(base_dir=str(tmp_path))
    content = b"id,name,val\n1,Alpha,10.5\n2,Beta,20.2\n3,Gamma,30.1\n"
    expected_hash = hashlib.sha256(content).hexdigest()
    
    file = UploadFile(filename="dataset.csv", file=io.BytesIO(content))
    file_ref, file_hash, total_bytes = await provider.save_upload_streaming(
        file=file,
        filename="dataset.csv",
        max_bytes=1024 * 1024
    )
    
    assert file_hash == expected_hash
    assert total_bytes == len(content)
    assert os.path.exists(file_ref)
    
    with open(file_ref, "rb") as f:
        saved_content = f.read()
    assert saved_content == content


@pytest.mark.anyio
async def test_single_pass_streaming_size_limit_exceeded(tmp_path):
    """Verify stream terminates and cleans up immediately when size limit is exceeded."""
    provider = LocalStorageProvider(base_dir=str(tmp_path))
    large_content = b"x" * 20000
    
    file = UploadFile(filename="large.csv", file=io.BytesIO(large_content))
    with pytest.raises(HTTPException) as exc_info:
        await provider.save_upload_streaming(
            file=file,
            filename="large.csv",
            max_bytes=1000  # 1 KB limit
        )
    
    assert exc_info.value.status_code == 413
    # Verify no lingering file remained in storage directory
    assert len(list(tmp_path.iterdir())) == 0


@pytest.mark.anyio
async def test_single_pass_streaming_magic_byte_validation(tmp_path):
    """Verify single-pass streaming validates file header signatures on chunk 0."""
    provider = LocalStorageProvider(base_dir=str(tmp_path))
    
    # 1. Spoofed Parquet (invalid signature)
    invalid_parquet = UploadFile(filename="fake.parquet", file=io.BytesIO(b"NOTPARQUETCONTENT123"))
    with pytest.raises(HTTPException) as exc_info:
        await provider.save_upload_streaming(
            file=invalid_parquet,
            filename="fake.parquet",
            max_bytes=1024 * 1024
        )
    assert exc_info.value.status_code == 400
    assert "Parquet header signature missing" in exc_info.value.detail

    # 2. Valid Parquet signature
    valid_parquet = UploadFile(filename="real.parquet", file=io.BytesIO(b"PAR1" + b"\x00" * 20))
    file_ref, file_hash, total_bytes = await provider.save_upload_streaming(
        file=valid_parquet,
        filename="real.parquet",
        max_bytes=1024 * 1024
    )
    assert total_bytes == 24
    assert os.path.exists(file_ref)
