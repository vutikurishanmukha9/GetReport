import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add Backend to path
sys.path.append(os.path.join(os.getcwd(), "Backend"))

from app.core.config import settings
from app.services.rag_service import EnhancedRAGService

def test_gemini_settings_and_rag_initialization():
    orig_gemini = settings.GEMINI_API_KEY
    orig_google = settings.GOOGLE_API_KEY
    orig_openrouter = settings.OPENROUTER_API_KEY
    orig_openai = settings.OPENAI_API_KEY

    try:
        # 1. Test GEMINI_API_KEY priority
        settings.GEMINI_API_KEY = "AIzaSyFakeGeminiKey"
        settings.GOOGLE_API_KEY = None
        settings.OPENROUTER_API_KEY = None
        settings.OPENAI_API_KEY = None

        rag = EnhancedRAGService()
        assert rag.enabled is True
        assert rag._provider_name == "Google Gemini"
        assert rag._base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
        assert rag.embedding_model == "text-embedding-004"
        assert "gemini-2.5-flash" in rag._models

        # 2. Test GOOGLE_API_KEY alias
        settings.GEMINI_API_KEY = None
        settings.GOOGLE_API_KEY = "AIzaSyFakeGoogleKey"
        rag_google = EnhancedRAGService()
        assert rag_google.enabled is True
        assert rag_google._provider_name == "Google Gemini"
        assert rag_google.api_key == "AIzaSyFakeGoogleKey"

    finally:
        settings.GEMINI_API_KEY = orig_gemini
        settings.GOOGLE_API_KEY = orig_google
        settings.OPENROUTER_API_KEY = orig_openrouter
        settings.OPENAI_API_KEY = orig_openai
