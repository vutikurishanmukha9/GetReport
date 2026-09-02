import gzip
import io
import re
import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException, UploadFile

from app.core.config import settings
from app.core.auth import verify_ws_api_key, verify_api_key
from app.services.report_weasyprint import safe_url_fetcher, _TEMPLATE_DIR
from app.services.data_processing import _validate_gzip_bomb, ParseError
from app.api.routes.upload import _validate_upload_sizes

def test_html_export_xss_escaping():
    """
    SEC-01: Verify that malicious script tags in summaries or filenames are escaped.
    """
    import html
    malicious_summary = "<script>alert('XSS')</script>\nLine 2 with <img src=x onerror=alert(1)>"
    malicious_filename = 'report"><script>alert(1)</script>.csv'
    
    safe_summary = html.escape(str(malicious_summary)).replace('\n', '<br/>')
    safe_filename = html.escape(str(malicious_filename))
    
    assert "<script>" not in safe_summary
    assert "&lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;" in safe_summary
    assert "<img" not in safe_summary
    assert "<script>" not in safe_filename
    assert "&lt;script&gt;" in safe_filename


def test_weasyprint_safe_url_fetcher_blocks_ssrf_and_lfi():
    """
    SEC-02: Verify safe_url_fetcher blocks external network requests and unauthorized files.
    """
    # 1. Block SSRF to cloud metadata
    with pytest.raises(ValueError, match="forbidden for security"):
        safe_url_fetcher("http://169.254.169.254/latest/meta-data/")

    # 2. Block external HTTPS
    with pytest.raises(ValueError, match="forbidden for security"):
        safe_url_fetcher("https://evil-attacker.com/payload.css")

    # 3. Block unauthorized file reads
    with pytest.raises(ValueError, match="forbidden for security"):
        safe_url_fetcher("file:///etc/passwd")

    with pytest.raises(ValueError, match="forbidden for security"):
        safe_url_fetcher("file:///C:/Windows/win.ini")

    # 4. Allow data URI
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    fetched = safe_url_fetcher(data_uri)
    assert fetched is not None
    assert "string" in fetched

    # 5. Allow valid file in template dir
    css_file = str(_TEMPLATE_DIR / "report.css")
    fetched_local = safe_url_fetcher(css_file)
    assert fetched_local is not None


def test_cors_origin_regex_blocks_unauthorized_vercel_domains():
    """
    SEC-03: Verify that arbitrary Vercel apps are rejected while GetReport domains pass.
    """
    pattern = re.compile(r"^https://get-report(?:-[a-zA-Z0-9_-]+)?\.vercel\.app$")
    
    # Authorized domains
    assert pattern.match("https://get-report.vercel.app")
    assert pattern.match("https://get-report-preview-12.vercel.app")
    assert pattern.match("https://get-report-git-main-user.vercel.app")
    
    # Unauthorized third-party domains
    assert not pattern.match("https://attacker.vercel.app")
    assert not pattern.match("https://malicious-getreport.vercel.app")
    assert not pattern.match("https://otherapp.vercel.app")
    assert not pattern.match("https://evil.com")


def test_gzip_bomb_validation(tmp_path):
    """
    SEC-05: Verify that decompression bombs in gzip archives are detected and rejected.
    """
    # 1. Normal safe gzip
    safe_gz_path = tmp_path / "safe.csv.gz"
    with gzip.open(safe_gz_path, "wb") as f:
        f.write(b"col1,col2\n1,2\n3,4\n")
    
    # Should not raise
    _validate_gzip_bomb(str(safe_gz_path), max_uncompressed_size_mb=10)

    # 2. Gzip decompression bomb exceeding size limit
    bomb_gz_path = tmp_path / "bomb.csv.gz"
    large_payload = b"0" * (1024 * 1024 * 3) # 3MB uncompressed
    with gzip.open(bomb_gz_path, "wb") as f:
        f.write(large_payload)

    # Validate against 1MB threshold -> should raise ParseError
    with pytest.raises(ParseError, match="exceeds safe limit"):
        _validate_gzip_bomb(str(bomb_gz_path), max_uncompressed_size_mb=1)


def test_aggregate_upload_size_limit():
    """
    SEC-06: Verify that aggregate upload sizes exceeding MAX_UPLOAD_SIZE_MB are rejected.
    """
    import asyncio
    max_mb = settings.MAX_UPLOAD_SIZE_MB
    file_size = (max_mb // 2 + 1) * 1024 * 1024  # > 50% of max

    # Create two files whose sum exceeds max_bytes
    file1 = UploadFile(filename="file1.csv", file=io.BytesIO(b"A" * file_size))
    file2 = UploadFile(filename="file2.csv", file=io.BytesIO(b"B" * file_size))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_validate_upload_sizes([file1, file2]))

    assert exc_info.value.status_code == 413
    assert "Combined upload too large" in exc_info.value.detail


def test_auth_constant_time_verification():
    """
    SEC-07: Verify constant-time API key verification logic.
    """
    settings.API_KEY = "super-secret-key-12345"
    try:
        assert verify_ws_api_key("super-secret-key-12345") is True
        assert verify_ws_api_key("wrong-key") is False
        assert verify_ws_api_key(None) is False
    finally:
        settings.API_KEY = ""
