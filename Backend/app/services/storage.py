import abc
import os
import shutil
import uuid
from pathlib import Path
from tempfile import mkstemp
from typing import BinaryIO, Optional

from app.core.config import settings

class StorageProvider(abc.ABC):
    """
    Abstract base class for file storage (Local, S3, Azure, etc.)
    """

    @abc.abstractmethod
    def save_upload(self, file_obj: BinaryIO, filename: str) -> str:
        """
        Save an uploaded file and return a reference path/ID.
        """
        pass

    @abc.abstractmethod
    def get_absolute_path(self, file_ref: str) -> str:
        """
        Get absolute local path for processing.
        For S3, this might involve downloading to a temp file first.
        """
        pass
        
    @abc.abstractmethod
    def delete(self, file_ref: str) -> bool:
        """
        Delete the file.
        """
        pass

class LocalStorageProvider(StorageProvider):
    """
    Stores files on the local filesystem.
    Suitable for development or single-server deployment.
    """
    def __init__(self, base_dir: str = "temp_uploads"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, file_obj: BinaryIO, filename: str) -> str:
        # Create a unique filename to prevent collisions
        ext = Path(filename).suffix
        unique_name = f"{uuid.uuid4()}{ext}"
        target_path = self.base_dir / unique_name
        
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file_obj, buffer)
            
        return str(target_path)

    def get_absolute_path(self, file_ref: str) -> str:
        # In local storage, the ref is the path
        return os.path.abspath(file_ref)

    def delete(self, file_ref: str) -> bool:
        try:
            os.remove(file_ref)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

# ─── Factory ─────────────────────────────────────────────────────────────────

def get_storage_provider() -> StorageProvider:
    # Future: Switch based on settings.STORAGE_TYPE
    # if settings.STORAGE_TYPE == "s3": return S3StorageProvider()
    return LocalStorageProvider()
