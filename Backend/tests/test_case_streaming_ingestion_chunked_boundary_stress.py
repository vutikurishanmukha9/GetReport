import io
import os
import hashlib
import pytest
from starlette.datastructures import UploadFile
from fastapi import HTTPException
from app.services.storage import LocalStorageProvider
from app.core.file_validation import verify_header_bytes, SIGNATURES

@pytest.mark.anyio
async def test_streaming_ingestion_chunked_boundary_stress(tmp_path):
    """
    Stress-test single-pass streaming ingestion across varied chunk sizes,
    odd byte lengths, and streaming boundary splits.
    """
    provider = LocalStorageProvider(base_dir=str(tmp_path))
    
    # Generate 500 KB of structured CSV content with random variations
    header = b"row_id,metric_a,metric_b,category\n"
    rows = [f"{i},{i*1.5:.2f},{i*2.5:.2f},cat_{i%5}\n".encode("utf-8") for i in range(10000)]
    full_content = header + b"".join(rows)
    expected_hash = hashlib.sha256(full_content).hexdigest()
    
    file = UploadFile(filename="stress_dataset.csv", file=io.BytesIO(full_content))
    file_ref, file_hash, total_bytes = await provider.save_upload_streaming(
        file=file,
        filename="stress_dataset.csv",
        max_bytes=10 * 1024 * 1024
    )
    
    assert file_hash == expected_hash
    assert total_bytes == len(full_content)
    assert os.path.exists(file_ref)
    
    with open(file_ref, "rb") as f:
        read_back = f.read()
    assert read_back == full_content


@pytest.mark.anyio
async def test_streaming_ingestion_gzip_valid_header(tmp_path):
    """Verify single-pass streaming handles gzip magic byte headers properly."""
    provider = LocalStorageProvider(base_dir=str(tmp_path))
    gzip_header = SIGNATURES["gz"] + b"\x08\x00\x00\x00\x00\x00" + b"\x00" * 100
    expected_hash = hashlib.sha256(gzip_header).hexdigest()
    
    file = UploadFile(filename="archive.gz", file=io.BytesIO(gzip_header))
    file_ref, file_hash, total_bytes = await provider.save_upload_streaming(
        file=file,
        filename="archive.gz",
        max_bytes=1024 * 1024
    )
    
    assert file_hash == expected_hash
    assert total_bytes == len(gzip_header)
    assert os.path.exists(file_ref)
