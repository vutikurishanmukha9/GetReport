import sys
import os

# Add Backend to path
sys.path.append(os.path.join(os.getcwd(), "Backend"))

from app.services.storage import get_storage_provider, LocalStorageProvider
try:
    from app.services.storage import S3StorageProvider
    print("S3StorageProvider class found.")
except ImportError:
    print("S3StorageProvider class NOT found.")

def test_factory():
    provider = get_storage_provider()
    print(f"Factory returned: {type(provider).__name__}")
    
    if isinstance(provider, LocalStorageProvider):
        print("PASS: Default provider is LocalStorageProvider")
    else:
        print("FAIL: Default provider is NOT LocalStorageProvider")

if __name__ == "__main__":
    test_factory()
