import io
import os
import pytest
from app.services.storage import DatabaseStorageProvider

def test_database_storage_provider_lifecycle():
    """Verify DatabaseStorageProvider saves file bytes to DB and retrieves them correctly."""
    provider = DatabaseStorageProvider()
    content = b"COL_A,COL_B\n10,20\n30,40"
    file_obj = io.BytesIO(content)
    
    file_ref = provider.save_upload(file_obj, "test_dataset.csv")
    assert file_ref.endswith(".csv")
    
    abs_path = provider.get_absolute_path(file_ref)
    assert os.path.exists(abs_path)
    with open(abs_path, "rb") as f:
        assert f.read() == content
        
    assert provider.delete(file_ref) is True
