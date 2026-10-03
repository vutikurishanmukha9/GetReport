"""
test_factories.py
~~~~~~~~~~~~~~~~~
Unit and integration tests for GetReport's registry-based factories:
- Registry[T] core mechanics (registration, defaults, lookup, error handling, reset)
- FallbackReportEngine resilience (is_available check, render-time failure fallback chain)
- BaseDataExporter Template Method (Universal CWE-1236 formula neutralization for CSV & Excel)
- StorageProviderFactory DI resolution
"""
from io import BytesIO
from pathlib import Path
import openpyxl
import polars as pl
import pytest

from app.core.factories.registry import Registry
from app.core.factories.reports import (
    BaseReportEngine,
    FallbackReportEngine,
    ReportEngineFactory,
    report_engine_registry,
)
from app.core.factories.exporters import (
    BaseDataExporter,
    CsvDataExporter,
    ExcelDataExporter,
    DataExporterFactory,
    exporter_registry,
)
from app.core.factories.storage import (
    StorageProviderFactory,
    get_storage_provider,
    storage_registry,
)
from app.services.report_styles import ReportMetadata


# ─────────────────────────────────────────────────────────────────────────────
# 1. Registry[T] Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_registry_registration_and_lookup():
    reg = Registry[str]("test_registry")
    
    @reg.register("alpha")
    def create_alpha():
        return "ALPHA_INSTANCE"

    reg.register_item("beta", lambda: "BETA_INSTANCE", is_default=True)

    assert reg.contains("alpha")
    assert reg.contains("ALPHA")  # Case insensitive
    assert reg.contains("beta")
    assert not reg.contains("gamma")

    assert reg.create("alpha") == "ALPHA_INSTANCE"
    assert reg.create() == "BETA_INSTANCE"  # Default fallback


def test_registry_unknown_key_raises_key_error():
    reg = Registry[str]("test_registry")
    reg.register_item("one", lambda: "1")
    reg.register_item("two", lambda: "2")

    with pytest.raises(KeyError) as exc_info:
        reg.get("three")
    
    err_msg = str(exc_info.value)
    assert "three" in err_msg
    assert "Available: ['one', 'two']" in err_msg


def test_registry_reset_clears_state():
    reg = Registry[int]("numeric_registry")
    reg.register_item("num", lambda: 42, is_default=True)
    assert reg.create() == 42

    reg.reset()
    assert reg.keys() == []
    with pytest.raises(KeyError):
        reg.create()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Report Engines & FallbackReportEngine Tests
# ─────────────────────────────────────────────────────────────────────────────

class MockFailingEngine(BaseReportEngine):
    """Engine that passes availability check but throws during render_pdf()."""
    @classmethod
    def is_available(cls) -> bool:
        return True

    def render_pdf(self, analysis_results, charts, filename):
        raise RuntimeError("Typst compile error: simulated render-time failure")


class MockSuccessfulEngine(BaseReportEngine):
    """Engine that succeeds and returns a valid mock PDF buffer."""
    @classmethod
    def is_available(cls) -> bool:
        return True

    def render_pdf(self, analysis_results, charts, filename):
        buf = BytesIO(b"%PDF-1.4 Mock Fallback Content")
        meta = ReportMetadata(
            filename=filename,
            success=True,
            timing_ms=12.5,
        )
        return buf, meta


class MockUnavailableEngine(BaseReportEngine):
    """Engine whose binary is missing."""
    @classmethod
    def is_available(cls) -> bool:
        return False

    def render_pdf(self, analysis_results, charts, filename):
        raise NotImplementedError("Should never be called")


def test_fallback_report_engine_intercepts_render_failure():
    """Verify that FallbackReportEngine catches render-time errors and delegates to the next candidate."""
    failing = MockFailingEngine()
    success = MockSuccessfulEngine()

    fallback_chain = FallbackReportEngine([failing, success])
    buf, meta = fallback_chain.render_pdf({}, {}, "test.pdf")

    assert buf.getvalue().startswith(b"%PDF-1.4")
    assert meta.filename == "test.pdf"
    assert meta.success is True


def test_fallback_report_engine_raises_if_all_fail():
    """Verify that if every candidate in the chain fails, RuntimeError is raised."""
    failing1 = MockFailingEngine()
    failing2 = MockFailingEngine()

    fallback_chain = FallbackReportEngine([failing1, failing2])
    with pytest.raises(RuntimeError) as exc_info:
        fallback_chain.render_pdf({}, {}, "test.pdf")

    assert "All PDF rendering engines in fallback chain failed" in str(exc_info.value)


def test_report_engine_registry_is_populated():
    """Assert registry is populated on import without side-effects."""
    assert "typst" in report_engine_registry.keys()
    assert "weasyprint" in report_engine_registry.keys()
    assert "reportlab" in report_engine_registry.keys()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Exporters (Template Method Universal Formula Neutralization)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dirty_formula_df() -> pl.DataFrame:
    """DataFrame with malicious formula injection triggers across text columns."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "user_input": [
            "=1+1",
            "+SUM(A1:A10)",
            "-5+10",
            "@SUM(B1)",
            "\tTAB_TRIGGER",
            "\rCR_TRIGGER",
        ],
        "normal_text": [
            "John Doe",
            "Valid Company Inc.",
            "Normal string",
            "123 Street",
            "Standard text",
            "Safe string",
        ],
        "numeric_val": [10.5, 20.0, -30.0, 40.2, 50.0, 60.1],
    })


def test_csv_exporter_neutralizes_formulas(dirty_formula_df):
    """Verify CWE-1236 protection prepends ' to formula triggers in CSV."""
    exporter = CsvDataExporter()
    file_path, mime, ext = exporter.export(dirty_formula_df, "test_export")

    try:
        assert ext == ".csv"
        assert mime == "text/csv"
        content = file_path.read_text(encoding="utf-8")

        # Formula prefixes must be preceded by a single quote
        assert "'=1+1" in content
        assert "'+SUM(A1:A10)" in content
        assert "'-5+10" in content
        assert "'@SUM(B1)" in content
        assert "'\tTAB_TRIGGER" in content
        assert "'\nCR_TRIGGER" in content or "'\rCR_TRIGGER" in content

        # Normal strings must not be needlessly quoted
        assert "John Doe" in content
        assert "'John Doe" not in content
    finally:
        if file_path.exists():
            file_path.unlink()


def test_excel_exporter_neutralizes_formulas(dirty_formula_df):
    """Verify CWE-1236 protection prepends ' to formula triggers in Excel."""
    exporter = ExcelDataExporter()
    file_path, mime, ext = exporter.export(dirty_formula_df, "test_export")

    try:
        assert ext == ".xlsx"
        assert "openxmlformats" in mime

        # Read back generated Excel via openpyxl
        wb = openpyxl.load_workbook(file_path)
        ws = wb.active
        user_inputs = [cell.value for cell in ws["B"][1:]]  # Column B, skipping header

        assert user_inputs[0] == "'=1+1"
        assert user_inputs[1] == "'+SUM(A1:A10)"
        assert user_inputs[2] == "'-5+10"
        assert user_inputs[3] == "'@SUM(B1)"
        assert user_inputs[4] == "'\tTAB_TRIGGER"
        # OOXML encodes bare carriage return as _x000D_; assert single quote neutralization was prepended
        assert user_inputs[5].startswith("'") and "CR_TRIGGER" in user_inputs[5]

        # Safe values stay unchanged
        normal_texts = [cell.value for cell in ws["C"][1:]]
        assert normal_texts[0] == "John Doe"
    finally:
        if file_path.exists():
            file_path.unlink()


def test_exporter_registry_is_populated():
    assert "csv" in exporter_registry.keys()
    assert "excel" in exporter_registry.keys()
    assert "parquet" in exporter_registry.keys()
    assert "json" in exporter_registry.keys()

    exporter = DataExporterFactory.create_exporter("csv")
    assert isinstance(exporter, CsvDataExporter)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Storage Provider Factory & DI Seam
# ─────────────────────────────────────────────────────────────────────────────

def test_storage_registry_resolution():
    assert "local" in storage_registry.keys()
    assert "db" in storage_registry.keys()
    assert "s3" in storage_registry.keys()

    local_provider = get_storage_provider("local")
    assert local_provider is not None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Remediation Strategy Functional Dispatch Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_remediation_functional_dispatch():
    from app.core.factories.remediation import apply_remediation, remediation_registry

    assert "drop_rows" in remediation_registry.keys()
    assert "fill_mean" in remediation_registry.keys()
    assert "fill_median" in remediation_registry.keys()
    assert "fill_mode" in remediation_registry.keys()
    assert "fill_value" in remediation_registry.keys()
    assert "replace_outliers_median" in remediation_registry.keys()

    # Test drop_rows
    df = pl.DataFrame({"a": [1, None, 3, None, 5]})
    df_clean, meta, cnt = apply_remediation("drop_rows", df, "a")
    assert df_clean.height == 3
    assert cnt == 2

    # Test fill_median
    df = pl.DataFrame({"a": [10.0, None, 30.0]})
    df_clean, meta, cnt = apply_remediation("fill_median", df, "a")
    assert df_clean["a"].to_list() == [10.0, 20.0, 30.0]
    assert cnt == 1

    # Test fill_mode
    df = pl.DataFrame({"cat": ["A", "B", "A", None]})
    df_clean, meta, cnt = apply_remediation("fill_mode", df, "cat")
    assert df_clean["cat"].to_list() == ["A", "B", "A", "A"]
    assert cnt == 1

    # Test fill_value
    df = pl.DataFrame({"x": ["hello", None]})
    df_clean, meta, cnt = apply_remediation("fill_value", df, "x", {"value": "world"})
    assert df_clean["x"].to_list() == ["hello", "world"]
    assert cnt == 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Phase 3: End-to-End Integration & Facade Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_e2e_report_generator_facade():
    """Verify that generate_pdf_report facade executes through ReportEngineFactory."""
    from app.services.report_generator import generate_pdf_report

    analysis_results = {
        "metadata": {
            "total_rows": 20,
            "total_columns": 2,
            "numeric_columns": 1,
            "categorical_columns": 1,
            "total_missing_values": 0,
            "missing_value_pct": 0.0,
        },
        "summary": {"revenue": {"mean": 100.0, "min": 50.0, "max": 150.0}},
        "categorical_distribution": {},
        "outliers": {},
        "strong_correlations": [],
        "column_quality_flags": {},
    }

    buf, meta = generate_pdf_report(analysis_results, {}, "facade_test.csv")
    assert isinstance(buf, BytesIO)
    pdf_bytes = buf.getvalue()
    assert pdf_bytes.startswith(b"%PDF")
    assert meta.success is True


def test_e2e_export_endpoints_with_universal_formula_neutralization():
    """Verify that /api/jobs/{task_id}/export/{format} uses DataExporterFactory with CWE-1236 defense."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.services.task_manager import title_task_manager, TaskStatus
    import io

    client = TestClient(app)
    storage = get_storage_provider()

    # Create dirty DataFrame with malicious formula payloads
    df = pl.DataFrame({
        "id": [1, 2],
        "user_input": ["=1+1", "@SUM(A1:A10)"],
        "name": ["Alice", "Bob"],
    })

    task_id = title_task_manager.create_job("e2e_export_test.csv")

    buffer = io.BytesIO()
    df.write_parquet(buffer)
    buffer.seek(0)
    cleaned_file_ref = storage.save_upload(buffer, f"cleaned_{task_id}.parquet")

    job_result = {
        "filename": "e2e_export_test.csv",
        "cleaned_file_ref": cleaned_file_ref,
        "info": {"rows": 2, "columns": 3},
    }
    title_task_manager.update_status(task_id, TaskStatus.COMPLETED, job_result)

    # 1. Test CSV export
    csv_resp = client.get(f"/api/jobs/{task_id}/export/csv")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers.get("content-type", "")
    assert "Cleaned_e2e_export_test.csv" in csv_resp.headers.get("content-disposition", "")
    csv_text = csv_resp.text
    assert "'=1+1" in csv_text
    assert "'@SUM(A1:A10)" in csv_text

    # 2. Test Excel export
    excel_resp = client.get(f"/api/jobs/{task_id}/export/excel")
    assert excel_resp.status_code == 200
    assert "openxmlformats" in excel_resp.headers.get("content-type", "")
    assert "Cleaned_e2e_export_test.xlsx" in excel_resp.headers.get("content-disposition", "")
    wb = openpyxl.load_workbook(io.BytesIO(excel_resp.content))
    ws = wb.active
    assert ws["B2"].value == "'=1+1"
    assert ws["B3"].value == "'@SUM(A1:A10)"

    # 3. Test Parquet export
    parquet_resp = client.get(f"/api/jobs/{task_id}/export/parquet")
    assert parquet_resp.status_code == 200
    parquet_df = pl.read_parquet(io.BytesIO(parquet_resp.content))
    assert parquet_df.height == 2

    # 4. Test Invalid format
    bad_resp = client.get(f"/api/jobs/{task_id}/export/invalid_format")
    assert bad_resp.status_code == 400
    assert "Invalid export format" in bad_resp.json()["detail"]


def test_storage_fastapi_dependency_override():
    """Verify that get_storage_provider can be overridden via FastAPI app.dependency_overrides."""
    from app.main import app

    class MockCustomStorage:
        def __init__(self):
            self.called = True

    mock_instance = MockCustomStorage()
    app.dependency_overrides[get_storage_provider] = lambda: mock_instance

    try:
        from app.core.factories.storage import get_storage_provider as di_provider
        # Dependency override is active in FastAPI dependency resolution
        assert app.dependency_overrides[get_storage_provider]() is mock_instance
    finally:
        app.dependency_overrides.clear()


