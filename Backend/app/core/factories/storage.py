"""
storage.py
~~~~~~~~~~
Storage Provider Factory powered by Registry[StorageProvider].
Provides clean dependency-injection seams for FastAPI endpoints and tests.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.core.config import settings
from app.core.factories.registry import Registry
from app.services.storage import (
    DatabaseStorageProvider,
    LocalStorageProvider,
    S3StorageProvider,
    StorageProvider,
)

logger = logging.getLogger(__name__)

storage_registry = Registry[StorageProvider]("storage_providers")
storage_registry.register_item("local", LocalStorageProvider, is_default=True)
storage_registry.register_item("db", DatabaseStorageProvider)
storage_registry.register_item("s3", S3StorageProvider)


class StorageProviderFactory:
    """Creator: Registry-based factory for storage backends."""

    @classmethod
    def create_provider(cls, storage_type: Optional[str] = None) -> StorageProvider:
        target = (storage_type or getattr(settings, "STORAGE_TYPE", "local") or "local").lower().strip()
        return storage_registry.create(target)


def get_storage_provider(storage_type: Optional[str] = None) -> StorageProvider:
    """
    Public entry point and FastAPI dependency-injection provider.
    Supports easy overriding in tests via app.dependency_overrides[get_storage_provider] = MockStorage.
    """
    return StorageProviderFactory.create_provider(storage_type)
