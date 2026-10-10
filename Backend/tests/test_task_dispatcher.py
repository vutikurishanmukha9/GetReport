"""
Integration tests for task dispatcher environment handling.
"""
import os
import pytest
from app.core import task_dispatcher
from app.core.config import settings


def test_check_redis_available_render_env(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(settings, "REDIS_URL", "redis://localhost:6379")
    task_dispatcher._redis_available = None

    result = task_dispatcher.check_redis_available()
    assert result is False


def test_check_redis_available_empty_url(monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setattr(settings, "REDIS_URL", "")
    task_dispatcher._redis_available = None

    result = task_dispatcher.check_redis_available()
    assert result is False
