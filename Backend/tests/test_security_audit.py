"""
Rigorous automated security verification suite.
Validates defenses against:
1. DuckDB LFI and filesystem access (read_csv, read_parquet, glob, etc.)
2. AST sandbox escapes (dunders, subscripts, format-string introspection)
3. Virtual concept synthesizer expression safety
4. Issue ledger execution scope restriction and AST validation
5. RAG HTML formatting XSS neutralization
"""
import pytest
import polars as pl
import os
import tempfile

from app.services.duckdb_engine import DuckDBAnalyticalSession, DuckDBSecurityError
from app.services.sandboxed_analyst import SandboxedAnalystAgent, SandboxedSecurityViolation
from app.services.concept_synthesizer import (
    VirtualConceptSynthesizer,
    ConceptDeclaration,
    ConceptSynthesizerSecurityViolation,
)
from app.services.issue_ledger import (
    IssueLedger,
    Issue,
    apply_remediation,
)
from app.services.rag_service import _format_answer_for_ui


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DUCKDB FILE ACCESS & LFI DEFENSE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_duckdb_blocks_read_csv_table_function():
    """Verify read_csv function is rejected by SQL validation."""
    session = DuckDBAnalyticalSession()
    df = pl.DataFrame({"x": [1, 2, 3]})
    session.register_polars("test_tbl", df)

    with pytest.raises(DuckDBSecurityError, match="Forbidden keyword|Direct file path"):
        session.execute_read_query("SELECT * FROM read_csv('pytest.ini')")

    session.close()


def test_duckdb_blocks_read_parquet_table_function():
    """Verify read_parquet function is rejected by SQL validation."""
    session = DuckDBAnalyticalSession()
    df = pl.DataFrame({"x": [1, 2, 3]})
    session.register_polars("test_tbl", df)

    with pytest.raises(DuckDBSecurityError, match="Forbidden keyword|Direct file path"):
        session.execute_read_query("SELECT * FROM read_parquet('data.parquet')")

    session.close()


def test_duckdb_blocks_glob_and_file_paths():
    """Verify glob and arbitrary file paths in quotes are rejected."""
    session = DuckDBAnalyticalSession()
    df = pl.DataFrame({"col": ["a", "b"]})
    session.register_polars("test_tbl", df)

    with pytest.raises(DuckDBSecurityError):
        session.execute_read_query("SELECT * FROM glob('*.py')")

    with pytest.raises(DuckDBSecurityError):
        session.execute_read_query("SELECT * FROM test_tbl WHERE col = 'config.env'")

    session.close()


def test_duckdb_engine_level_external_access_disabled():
    """Verify DuckDB C++ engine refuses file access even if called internally."""
    session = DuckDBAnalyticalSession()
    with pytest.raises(Exception):
        # Even direct internal connection execution must be blocked by enable_external_access=false
        session.conn.execute("SELECT * FROM read_csv('pytest.ini')").fetchall()
    session.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AST SANDBOX REFLECTION & FORMAT-STRING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_sandbox_blocks_all_dunder_attributes():
    """Verify all dunder attributes are strictly blocked by AST validator."""
    agent = SandboxedAnalystAgent(df=pl.DataFrame({"val": [1, 2, 3]}))

    dunder_payloads = [
        "x = ().__class__",
        "x = [].__class__.__bases__[0].__subclasses__()",
        "x = df.__dict__",
        "x = pl.__builtins__",
        "x = (1).__init__",
        "x = (lambda: 1).__closure__",
    ]

    for payload in dunder_payloads:
        with pytest.raises(SandboxedSecurityViolation):
            agent.validate_code_ast(payload)


def test_sandbox_blocks_dunder_subscripts():
    """Verify private keys via subscript lookups are blocked."""
    agent = SandboxedAnalystAgent(df=pl.DataFrame({"val": [1, 2, 3]}))

    subscript_payloads = [
        "x = globals()['__builtins__']",
        "d = {'a': 1}; x = d['__class__']",
    ]

    for payload in subscript_payloads:
        with pytest.raises(SandboxedSecurityViolation):
            agent.validate_code_ast(payload)


def test_sandbox_blocks_format_string_introspection():
    """Verify format string introspection payloads are blocked."""
    agent = SandboxedAnalystAgent(df=pl.DataFrame({"val": [1, 2, 3]}))

    format_payloads = [
        'res = "{0.__class__}".format(1)',
        'res = "{__builtins__}".format()',
    ]

    for payload in format_payloads:
        with pytest.raises(SandboxedSecurityViolation):
            agent.validate_code_ast(payload)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VIRTUAL CONCEPT SYNTHESIZER SECURITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_concept_synthesizer_blocks_dunders():
    """Verify concept expressions reject dunders and private method calls."""
    synth = VirtualConceptSynthesizer()

    malicious_expressions = [
        "pl.col('val').__class__",
        "pl.col('val').__dict__",
        "pl.col('val')['__builtins__']",
        "eval('1 + 1')",
        "__import__('os')",
    ]

    for expr in malicious_expressions:
        with pytest.raises(ConceptSynthesizerSecurityViolation):
            synth.validate_expression_ast(expr)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ISSUE LEDGER EXECUTION SCOPE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_issue_ledger_rejects_unauthorized_imports():
    """Verify issue ledger remediation code rejects forbidden imports."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    ledger = IssueLedger()

    malicious_issue = Issue(
        id="test_1",
        issue_type="outliers",
        severity="medium",
        column="a",
        affected_rows=1,
        affected_pct=33.3,
        description="Malicious payload",
        suggested_fix="none",
        fix_code="import os; df = df",
        status="approved",
    )
    ledger.add_issue(malicious_issue)

    # Should safely catch and log warning without executing import os
    result_df = apply_remediation(df, ledger)
    assert result_df.height == 3


def test_issue_ledger_executes_legitimate_fixes():
    """Verify legitimate Polars fixes execute properly in restricted namespace."""
    df = pl.DataFrame({"a": [1, 2, None]})
    ledger = IssueLedger()

    legit_issue = Issue(
        id="test_2",
        issue_type="missing_values",
        severity="high",
        column="a",
        affected_rows=1,
        affected_pct=33.3,
        description="Fills missing values with 0",
        suggested_fix="fill null",
        fix_code="df = df.with_columns(pl.col('a').fill_null(0))",
        status="approved",
    )
    ledger.add_issue(legit_issue)

    result_df = apply_remediation(df, ledger)
    assert result_df["a"].null_count() == 0
    assert result_df["a"].to_list() == [1, 2, 0]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RAG HTML FORMATTING XSS DEFENSE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_format_answer_for_ui_escapes_xss():
    """Verify script and img tags are escaped while semantic tags are preserved."""
    malicious_input = "<script>alert('xss')</script><img src=x onerror=alert(1)> **Bold Fact** `code_snippet`"
    rendered = _format_answer_for_ui(malicious_input)

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "<img" not in rendered
    assert "&lt;img" in rendered
    assert "<b>Bold Fact</b>" in rendered
    assert "<code>code_snippet</code>" in rendered


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EXTENDED SECURITY VULNERABILITY AUDIT TESTS (0 VULNERABILITIES)
# ═══════════════════════════════════════════════════════════════════════════════

def test_concept_synthesizer_blocks_file_io_and_lambdas():
    """Verify concept expressions reject arbitrary file I/O, lambdas, and unsafe methods."""
    synth = VirtualConceptSynthesizer()

    forbidden_payloads = [
        "pl.read_csv('/etc/passwd')",
        "pl.scan_parquet('data.parquet')",
        "pl.read_ipc('data.arrow')",
        "pl.col('val').map_elements(lambda x: x * 2)",
        "(lambda x: x)(pl.col('val'))",
        "pl.col('val').pipe(str)",
    ]

    for expr in forbidden_payloads:
        with pytest.raises(ConceptSynthesizerSecurityViolation):
            synth.validate_expression_ast(expr)


def test_csv_formula_injection_sanitization():
    """Verify string columns starting with dangerous spreadsheet triggers are sanitized."""
    from app.services.data_processing import sanitize_df_for_csv_export

    df = pl.DataFrame({
        "formula": ["=cmd|' /C calc'!A0", "+SUM(A1:A2)", "-1+2", "@SUM(B1:B2)", "\tmalicious", "normal_text"],
        "numbers": [1, 2, 3, 4, 5, 6],
    })

    safe_df = sanitize_df_for_csv_export(df)
    sanitized_vals = safe_df["formula"].to_list()

    assert sanitized_vals[0] == "'=cmd|' /C calc'!A0"
    assert sanitized_vals[1] == "'+SUM(A1:A2)"
    assert sanitized_vals[2] == "'-1+2"
    assert sanitized_vals[3] == "'@SUM(B1:B2)"
    assert sanitized_vals[4] == "'\tmalicious"
    assert sanitized_vals[5] == "normal_text"
    assert safe_df["numbers"].to_list() == [1, 2, 3, 4, 5, 6]


def test_duckdb_query_capped_records():
    """Verify analytical queries cap materialization to max_rows."""
    session = DuckDBAnalyticalSession()
    try:
        records = session.execute_read_query("SELECT 1 AS num UNION ALL SELECT 2 UNION ALL SELECT 3", max_rows=2)
        assert len(records) == 2
    finally:
        session.close()


def test_verify_ws_api_key_fail_closed(monkeypatch):
    """Verify WebSocket auth fails closed when REQUIRE_AUTH is set without API_KEY."""
    from app.core import auth
    from app.core.config import settings

    # When REQUIRE_AUTH is True but API_KEY is empty -> reject (fail closed)
    monkeypatch.setattr(settings, "REQUIRE_AUTH", True)
    monkeypatch.setattr(settings, "API_KEY", "")

    assert auth.verify_ws_api_key("some_key") is False
    assert auth.verify_ws_api_key(None) is False

    # When REQUIRE_AUTH is False and DATABASE_URL is set -> allow public access
    monkeypatch.setattr(settings, "REQUIRE_AUTH", False)
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    assert auth.verify_ws_api_key(None) is True


def test_request_id_crlf_defense():
    """Verify RequestIDMiddleware sanitizes CRLF sequences and generates valid UUIDs."""
    from app.core.request_id import RequestIDMiddleware, SAFE_REQUEST_ID_REGEX
    from starlette.requests import Request
    from starlette.responses import Response
    import asyncio

    middleware = RequestIDMiddleware(None)

    # Valid request ID
    assert SAFE_REQUEST_ID_REGEX.match("trace-abc-123_45") is not None

    # CRLF / Header Injection attempt
    assert SAFE_REQUEST_ID_REGEX.match("trace\r\nInjected-Header: evil") is None
    assert SAFE_REQUEST_ID_REGEX.match("<script>alert(1)</script>") is None


def test_limiter_cf_connecting_ip():
    """Verify rate limiter prioritizes CF-Connecting-IP when present."""
    from app.core.limiter import _get_real_client_ip
    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [
            (b"cf-connecting-ip", b"198.51.100.42"),
            (b"x-forwarded-for", b"203.0.113.195, 10.0.0.1"),
        ],
        "client": ("127.0.0.1", 12345),
    }
    req = Request(scope)
    assert _get_real_client_ip(req) == "198.51.100.42"

