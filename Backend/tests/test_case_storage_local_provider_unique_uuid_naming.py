import io
import os
import pytest
from app.services.storage import LocalStorageProvider

def test_storage_local_provider_generates_unique_filenames(tmp_path):
    """Verify multiple uploads with identical filenames receive collision-free unique UUID references."""
    storage = LocalStorageProvider(base_dir=str(tmp_path))
    
    f1 = io.BytesIO(b"Data version 1")
    f2 = io.BytesIO(b"Data version 2")
    
    path1 = storage.save_upload(f1, "dataset.csv")
    path2 = storage.save_upload(f2, "dataset.csv")
    
    assert path1 != path2
    assert os.path.exists(path1)
    assert os.path.exists(path2)
