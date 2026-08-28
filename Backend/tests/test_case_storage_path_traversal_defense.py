import io
import os
import pytest
from app.services.storage import LocalStorageProvider

def test_storage_path_traversal_defense_enforcement(tmp_path):
    """Verify LocalStorageProvider saves files properly and strips directory components."""
    storage = LocalStorageProvider(base_dir=str(tmp_path))
    
    # Save a normal file
    file_data = io.BytesIO(b"Safe dataset contents")
    saved_path = storage.save_upload(file_data, "data.csv")
    assert os.path.exists(saved_path)
    assert str(tmp_path) in saved_path
    
    # Test path resolution strips traversal elements
    resolved_path = storage.get_absolute_path("../../etc/passwd")
    assert str(tmp_path) in resolved_path
    assert "passwd" in resolved_path
    assert not resolved_path.startswith("/etc")
    
    # Clean up
    assert storage.delete(saved_path) is True
    assert not os.path.exists(saved_path)
