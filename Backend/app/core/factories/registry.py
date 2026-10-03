"""
registry.py
~~~~~~~~~~~
Generic, type-safe registry supporting decorator registration, default fallbacks,
and test-isolation resetting.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """
    Generic, inspectable registry for registering and instantiating components by key.
    Supports decorator registration, programmatic registration, default keys, and test resets.
    """

    def __init__(self, name: str):
        self.name = name
        self._entries: dict[str, type[T] | Callable[..., T]] = {}
        self._default_key: Optional[str] = None

    def register(self, key: str, is_default: bool = False):
        """Decorator to register a class or factory callable under a specific key."""
        def decorator(cls_or_func: type[T] | Callable[..., T]):
            self.register_item(key, cls_or_func, is_default=is_default)
            return cls_or_func
        return decorator

    def register_item(
        self,
        key: str,
        cls_or_func: type[T] | Callable[..., T],
        is_default: bool = False,
    ) -> None:
        """Programmatic registration."""
        normalized_key = key.lower().strip()
        self._entries[normalized_key] = cls_or_func
        if is_default:
            self._default_key = normalized_key

    def get(self, key: Optional[str] = None) -> type[T] | Callable[..., T]:
        """Look up constructor/class by key or fallback to default."""
        lookup_key = key or self._default_key
        if not lookup_key:
            raise KeyError(
                f"Registry '{self.name}' has no default key configured and none was provided."
            )
        normalized_key = lookup_key.lower().strip()
        if normalized_key not in self._entries:
            available = list(self._entries.keys())
            raise KeyError(
                f"Unknown key '{normalized_key}' in registry '{self.name}'. Available: {available}"
            )
        return self._entries[normalized_key]

    def create(self, key: Optional[str] = None, *args: Any, **kwargs: Any) -> T:
        """Instantiate or invoke the constructor registered under `key`."""
        constructor = self.get(key)
        return constructor(*args, **kwargs)

    def contains(self, key: str) -> bool:
        return key.lower().strip() in self._entries

    def keys(self) -> list[str]:
        return list(self._entries.keys())

    def reset(self) -> None:
        """Reset registry state. Crucial for unit test isolation."""
        self._entries.clear()
        self._default_key = None
