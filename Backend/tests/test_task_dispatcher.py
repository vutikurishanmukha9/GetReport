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


@pytest.mark.asyncio
async def test_dispatch_task_fails_closed_in_production(monkeypatch):
    from fastapi import HTTPException
    monkeypatch.setattr(task_dispatcher, "check_redis_available", lambda: False)
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    monkeypatch.setattr(settings, "REQUIRE_DURABLE_JOBS", False)

    with pytest.raises(HTTPException) as exc_info:
        await task_dispatcher.dispatch_task("test_job", 123)
    assert exc_info.value.status_code == 503
    assert "Durable job execution is required" in exc_info.value.detail


def test_dispatch_task_sync_fails_closed_when_durable_required(monkeypatch):
    monkeypatch.setattr(task_dispatcher, "check_redis_available", lambda: False)
    monkeypatch.setattr(settings, "REQUIRE_DURABLE_JOBS", True)

    with pytest.raises(RuntimeError) as exc_info:
        task_dispatcher.dispatch_task_sync("test_job", 123)
    assert "Durable job execution is required" in str(exc_info.value)

