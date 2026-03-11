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

class S3StorageProvider(StorageProvider):
    """
    Stores files in AWS S3.
    Requires: boto3
    """
    def __init__(self):
        try:
            import boto3
            self.s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            self.bucket = settings.AWS_BUCKET_NAME
        except ImportError:
            raise ImportError("boto3 is required for S3StorageProvider. pip install boto3")

    def save_upload(self, file_obj: BinaryIO, filename: str) -> str:
        # Create a unique key
        ext = os.path.splitext(filename)[1]
        unique_key = f"uploads/{uuid.uuid4()}{ext}"
        
        # Upload
        try:
            # Check if file_obj is bytes or specific wrapper
            # file_obj from UploadFile is SpooledTemporaryFile
            file_obj.seek(0)
            self.s3.upload_fileobj(file_obj, self.bucket, unique_key)
            return unique_key
        except Exception as e:
            raise RuntimeError(f"S3 Upload failed: {e}")

    def get_absolute_path(self, file_ref: str) -> str:
        # For processing, we often need a local file.
        # Check if we have it cached in /tmp, else download.
        local_cache_dir = Path("temp_cache")
        local_cache_dir.mkdir(exist_ok=True)
        
        local_path = local_cache_dir / os.path.basename(file_ref)
        
        if not local_path.exists():
            try:
                self.s3.download_file(self.bucket, file_ref, str(local_path))
            except Exception as e:
                raise RuntimeError(f"S3 Download failed: {e}")
                
        return str(local_path.absolute())

    def delete(self, file_ref: str) -> bool:
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=file_ref)
            return True
        except Exception:
            return False

class DatabaseStorageProvider(StorageProvider):
    """
    Stores files in the Database (Postgres BYTEA or SQLite BLOB).
    Suitable for PAAS deployments like Render with isolated filesystems (no shared disks).
    """
    def __init__(self):
        # Local cache for processing tools (fastexcel, polars)
        self.local_cache_dir = Path("temp_cache")
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, file_obj: BinaryIO, filename: str) -> str:
        from app.db import get_db_connection
        from app.core.config import settings as _settings
        
        ext = Path(filename).suffix
        unique_name = f"{uuid.uuid4()}{ext}"
        
        # Read the file into memory (FastAPI has already enforced size limits)
        file_obj.seek(0)
        raw_data = file_obj.read()
        
        # PostgreSQL BYTEA requires psycopg2.Binary() wrapping
        if _settings.DATABASE_URL:
            try:
                import psycopg2
                data = psycopg2.Binary(raw_data)
            except ImportError:
                data = raw_data
        else:
            data = raw_data
        
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO file_store (file_ref, data) VALUES (?, ?)",
                (unique_name, data)
            )
            conn.commit()
            
        return unique_name

    def get_absolute_path(self, file_ref: str) -> str:
        from app.db import get_db_connection
        local_path = self.local_cache_dir / os.path.basename(file_ref)
        
        if not local_path.exists():
            with get_db_connection() as conn:
                cursor = conn.execute(
                    "SELECT data FROM file_store WHERE file_ref = ?", 
                    (file_ref,)
                )
                row = cursor.fetchone()
                
                if not row:
                    raise FileNotFoundError(f"File {file_ref} not found in database storage")
                
                # Extract binary data — handle dict (psycopg2 RealDictCursor), 
                # sqlite3.Row, or tuple
                if isinstance(row, dict):
                    data = row['data']
                elif hasattr(row, 'keys'):
                    data = row['data']
                else:
                    data = row[0]
                    
                with open(local_path, "wb") as f:
                    if isinstance(data, memoryview):
                        f.write(bytes(data))
                    elif isinstance(data, bytes):
                        f.write(data)
                    else:
                        f.write(bytes(data))
                        
        return str(local_path.absolute())

    def delete(self, file_ref: str) -> bool:
        from app.db import get_db_connection
        # Delete from local cache
        local_path = self.local_cache_dir / os.path.basename(file_ref)
        if local_path.exists():
            try: local_path.unlink()
            except: pass
            
        # Delete from DB
        try:
            with get_db_connection() as conn:
                conn.execute("DELETE FROM file_store WHERE file_ref = ?", (file_ref,))
                conn.commit()
            return True
        except Exception:
            return False

# ─── Factory ─────────────────────────────────────────────────────────────────

def get_storage_provider() -> StorageProvider:
    if settings.STORAGE_TYPE.lower() == "s3":
        return S3StorageProvider()
    elif settings.STORAGE_TYPE.lower() == "db":
        return DatabaseStorageProvider()
    return LocalStorageProvider()
