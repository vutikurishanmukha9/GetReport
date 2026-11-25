"""
Regression & integration tests covering the resolved backend bridge gaps (BG-01..BG-08).
"""
from __future__ import annotations

import base64
import json
import uuid
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import polars as pl
import pytest
from fastapi.testclient import TestClient

from app.main import app


# ─── BG-01: Issue Ledger Decisions -> Cleaning & API Response Shape ───────────


def test_bg01_issue_ledger_from_dict_audit_and_rejected_skips():
    from app.services.data_processing import clean_data
    from app.services.issue_ledger import Issue, IssueLedger, apply_remediation_with_audit
    from app.services.transformation_dag import create_dag

    df = pl.DataFrame({
        "Approved Col": [10.0, None, 30.0, 10.0],
        "Rejected Col": [1.0, None, 3.0, 1.0],
    })
    ledger = IssueLedger(issues=[
        Issue(
            id="iss-app",
            issue_type="missing_values",
            severity="high",
            column="Approved Col",
            affected_rows=1,
            affected_pct=25.0,
            description="Null in Approved Col",
            suggested_fix="Fill with -99",
            fix_code="df = df.with_columns(pl.col('Approved Col').fill_null(-99.0))",
            status="modified",
            user_note="Use explicit sentinel",
        ),
        Issue(
            id="iss-rej",
            issue_type="missing_values",
            severity="medium",
            column="Rejected Col",
            affected_rows=1,
            affected_pct=25.0,
            description="Null in Rejected Col",
            suggested_fix="Fill median",
            fix_code="df = df.with_columns(pl.col('Rejected Col').fill_null(2.0))",
            status="rejected",
            user_note="Keep missing",
        ),
        Issue(
            id="iss-dup",
            issue_type="duplicates",
            severity="low",
            column=None,
            affected_rows=1,
            affected_pct=25.0,
            description="Duplicate rows",
            suggested_fix="Drop duplicates",
            fix_code="df = df.unique(maintain_order=True)",
            status="rejected",
            user_note="Keep duplicates intentionally",
        ),
    ])

    reconstructed = IssueLedger.from_dict(ledger.to_dict())
    assert len(reconstructed.issues) == 3
    assert reconstructed.issues[0].status == "modified"
    assert reconstructed.issues[1].status == "rejected"

    dag = create_dag(df, "bg01_test.csv")
    rem_df, audit = apply_remediation_with_audit(df, reconstructed, dag=dag)

    assert len(audit["applied"]) == 1
    assert audit["applied"][0]["id"] == "iss-app"
    assert audit["applied_columns"] == ["Approved Col"]
    assert len(audit["rejected"]) == 2
    assert rem_df["Approved Col"].null_count() == 0
    assert -99.0 in rem_df["Approved Col"].to_list()
    assert rem_df["Rejected Col"].null_count() == 1

    # Honor rejected column & rejected deduplication inside clean_data
    effective_rules = {
        "Approved Col": {"action": "skip"},
        "Rejected Col": {"action": "skip"},
        "skip_duplicates": True,
    }
    cleaned_df, report, final_dag = clean_data(rem_df, effective_rules, dag, "bg01_test.csv")
    assert cleaned_df.height == 4  # duplicate row preserved because skip_duplicates=True
    assert cleaned_df["rejected_col"].null_count() == 1  # null preserved due to skip
    assert -99.0 in cleaned_df["approved_col"].to_list()
    assert any(
        n.get("parameters", {}).get("source") == "issue_ledger"
        for n in final_dag.to_dict()["nodes"].values()
    )


def test_bg01_issue_endpoints_return_ledger_data(monkeypatch):
    from app.services.task_manager import TaskStatus, title_task_manager

    task_id = str(uuid.uuid4())
    fake_job = SimpleNamespace(
        task_id=task_id,
        owner_id=None,
        status=TaskStatus.WAITING_FOR_USER,
        result={
            "issue_ledger": {
                "issues": [
                    {
                        "id": "i1",
                        "issue_type": "missing_values",
                        "severity": "high",
                        "column": "amount",
                        "affected_rows": 5,
                        "affected_pct": 10.0,
                        "description": "Missing values",
                        "suggested_fix": "Fill median",
                        "fix_code": "df = df.with_columns(pl.col('amount').fill_null(0))",
                        "status": "pending",
                        "user_note": "",
                    }
                ],
                "summary": {"pending": 1, "approved": 0, "rejected": 0, "modified": 0, "total": 1},
                "locked": False,
                "locked_at": None,
            }
        },
    )

    monkeypatch.setattr(title_task_manager, "get_job_async", AsyncMock(return_value=fake_job))
    monkeypatch.setattr(title_task_manager, "update_result_async", AsyncMock(return_value=None))

    client = TestClient(app)
    resp = client.post(f"/api/jobs/{task_id}/issues/i1/approve", json={"note": "Looks good"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["approved"] == 1
    assert body["issues"][0]["status"] == "approved"
    assert body["locked"] is False
    assert body["message"] == "Issue approved"

    lock_resp = client.post(f"/api/jobs/{task_id}/issues/lock")
    assert lock_resp.status_code == 200
    assert lock_resp.json()["locked"] is True


# ─── BG-02 & BG-03: Zero Synthetic Fallbacks & Canonical Report Normalization ─


def test_bg02_bg03_no_synthetic_fallbacks_and_real_bridges(monkeypatch):
    from app.services.report_normalizer import normalize_analysis_for_reports
    from app.services.report_sections import _build_cleaning_section, _build_time_series_section, _build_visualizations
    from app.services.report_styles import ReportMetadata, get_custom_styles
    from app.services.report_typst import generate_pdf_typst
    from app.services.report_weasyprint import _JINJA_ENV

    minimal_analysis = {
        "metadata": {"total_rows": 120, "total_columns": 2, "missing_value_pct": 0.0},
        "summary": {
            "custom_metric": {"mean": 42.0, "median": 41.0, "std": 3.5, "min": 10.0, "max": 80.0, "missing_pct": 0.0}
        },
        "cleaning_report": {
            "duplicate_rows_removed": 3,
            "row_triage": {
                "summary": {
                    "by_reason": {
                        "blank_row": 2,
                        "repeated_header": 1,
                        "summary_row": 4,
                        "footer_note": 1,
                        "exact_duplicate": 3,
                    }
                }
            },
        },
        "time_series_analysis": {
            "has_time_series": True,
            "time_column": "event_timestamp",
            "mixed_date_formats": False,
            "trends": {
                "custom_metric": {
                    "direction": "stable",
                    "slope": 0.0,
                    "p_value": 0.85,
                    "z_score": 0.12,
                }
            },
            "seasonality": {},
        },
    }

    normalized = normalize_analysis_for_reports(minimal_analysis)
    serialized = json.dumps(normalized)
    for forbidden in ("1513", "credit_card", "bonus_payout", "churned", "order_date", "weight_kg", "87.28", "182"):
        assert forbidden not in serialized

    # Verify ReportLab cleaning section reads real row_intelligence codes + zero fake 1513/182
    styles = get_custom_styles()
    meta = ReportMetadata(filename="clean_custom.csv")
    cleaning_flow = _build_cleaning_section(minimal_analysis, styles, meta)
    assert cleaning_flow
    cleaning_text = " ".join(str(row) for fl in cleaning_flow if hasattr(fl, "_cellvalues") for row in fl._cellvalues)
    assert "1,513" not in cleaning_text and "182" not in cleaning_text
    assert "3" in cleaning_text and "2" in cleaning_text
    ts_flow = _build_time_series_section(minimal_analysis, styles, meta)
    assert ts_flow
    ts_text = " ".join(str(row) for fl in ts_flow if hasattr(fl, "_cellvalues") for row in fl._cellvalues)
    assert "Stable" in ts_text and "1.96" not in ts_text

    # Verify ReportLab renders charts["donut_chart"]
    from PIL import Image
    png_io = BytesIO()
    Image.new("RGB", (10, 10), color=(70, 130, 180)).save(png_io, format="PNG")
    tiny_png_b64 = base64.b64encode(png_io.getvalue()).decode("ascii")
    viz_flow = _build_visualizations({"donut_chart": {"column": "segment", "image": f"data:image/png;base64,{tiny_png_b64}"}}, styles, meta)
    assert viz_flow and "Visualizations" in meta.sections_included

    # Full canonical producer mapping (Benford, symbolic_equations, personas, text_hazards, PII dict, dict recs)
    rich_analysis = {
        **minimal_analysis,
        "confidence_scores": {
            "dataset_overall": 91.5,
            "grade": "A",
            "columns": [
                {
                    "column": "custom_metric",
                    "overall": 91.5,
                    "grade": "A",
                    "benford_analysis": {"applicable": True, "chi_square": 4.5, "mad": 0.008, "deviation_level": "Close conformity"},
                }
            ],
        },
        "smart_schema": {
            "symbolic_equations": [{"formula": "net = gross - tax", "confidence": 99.5, "note": "Verified arithmetic rule"}],
            "type_candidates": [{"column": "id_str", "current_type": "Utf8", "suggested_type": "Categorical", "confidence": 0.9}],
            "relationships": [{"type": "functional", "source": "gross", "target": "net", "confidence": 0.95}],
        },
        "unsupervised_learning": {"personas": [{"name": "Enterprise Buyers", "size": 50, "pct": 41.7, "summary": "High spend"}]},
        "supervised_learning": {"target": "custom_metric", "task": "regression", "model": "Ridge", "metrics": {"r2": 0.91}},
        "text_hazards": {"has_hazards": True, "by_type": {"xss": 2}},
        "pii": {
            "has_pii": True,
            "findings": {"contact_email": {"column": "contact_email", "pii_type": "EMAIL", "confidence": "HIGH", "sample_masked": "a***@x.com"}},
        },
        "recommendations": {"data_quality": [{"title": "Standardize units", "description": "Normalize currency"}]},
        "semantic_analysis": {
            "domain": {"primary": "finance", "confidence": 0.9},
            "column_roles": {"custom_metric": {"role": "metric"}},
            "suggested_analysis": [{"col_a": "gross", "col_b": "net", "reason": "Linear dependency"}],
        },
        "missing_patterns": {"inferred_pattern": "MCAR", "column_details": {"notes": {"count": 5, "percentage": 4.1}}},
    }

    rich_norm = normalize_analysis_for_reports(rich_analysis)
    assert rich_norm["benford"]["columns"][0]["column"] == "custom_metric"
    assert rich_norm["symbolic_equations"][0]["equation"] == "net = gross - tax"
    assert rich_norm["feature_engineering"]["clusters"][0]["name"] == "Enterprise Buyers"
    assert rich_norm["security_scan"]["findings"][0]["detections"] == 2
    assert rich_norm["recommendations"]["items"][0]["title"] == "Standardize units"
    assert rich_norm["pii"]["findings"][0]["entity"] == "EMAIL"

    # Typst end-to-end compilation without fake demo data
    pdf_buf, typst_meta = generate_pdf_typst(rich_analysis, {}, "real_custom.csv")
    assert typst_meta.success
    assert pdf_buf.getvalue().startswith(b"%PDF-")

    # WeasyPrint HTML Jinja template renders trends, missing_patterns, smart_schema, and semantic pairs
    html = _JINJA_ENV.get_template("report.html").render(
        filename="real_custom.csv",
        generated_at="now",
        metadata=rich_norm["metadata"],
        analysis=rich_norm,
        charts={},
    )
    assert "Standardize units" in html
    assert "gross" in html and "net" in html
    assert "stable" in html


# ─── BG-04: RAG, Chunker, & Knowledge Graph Production Alignment ──────────────


def test_bg04_graph_chunker_and_rag_qa_accept_production_schema():
    from app.core.rag_utils import TableSemanticChunker
    from app.services.dataset_graph_builder import build_dataset_graph
    from app.services.rag_service import _generate_smart_dataset_answer

    prod_job_result = {
        "filename": "orders.csv",
        "info": {
            "rows": 50,
            "columns": ["order_total", "region"],
            "dtypes": {"order_total": "Float64", "region": "Utf8"},
            "missing_values": {"order_total": {"percentage": 2.0}, "region": {"percentage": 0.0}},
        },
        "cleaning_report": {"total_changes": 2, "duplicate_rows_removed": 1},
        "issue_ledger": {
            "issues": [
                {
                    "id": "i-1",
                    "issue_type": "missing_values",
                    "severity": "medium",
                    "column": "region",
                    "description": "Missing region labels",
                    "suggested_fix": "Fill with Unknown",
                    "status": "pending",
                }
            ]
        },
        "analysis": {
            "metadata": {"total_rows": 50, "total_columns": 2, "missing_value_pct": 1.0},
            "summary": {"order_total": {"mean": 120.5, "min": 10.0, "max": 500.0, "std": 45.0, "null_count": 1}},
            "categorical_distribution": {"region": {"unique_count": 3, "top_values": [{"value": "East", "count": 20}]}},
            "strong_correlations": [],
            "confidence_scores": {"dataset_overall": 88.0, "grade": "B", "columns": []},
            "semantic_analysis": {"domain": {"primary": "sales_ecommerce", "confidence": 0.92}, "column_roles": {}},
            "recommendations": {
                "analysis": [{"title": "Segment by Region", "description": "Analyze regional variance", "priority": "high"}]
            },
            "pii": {"has_pii": True, "findings": {"region": {"column": "region", "pii_type": "ADDRESS", "confidence": "MEDIUM"}}},
            "data_quality": {"results": [{"test_type": "not_null", "column": "order_total", "success": False}]},
        },
    }

    store = build_dataset_graph("t-bg04", prod_job_result)
    assert store.graph.has_node("col:region")  # categorical column included!
    assert store.graph.has_node("issue:i-1")
    assert store.graph.has_node("pii:region")
    assert store.graph.has_node("dq_failure:order_total:not_null")
    ds_attrs = store.graph.nodes["dataset:orders.csv"]
    assert ds_attrs["domain"] == "sales_ecommerce"
    assert ds_attrs["confidence_score"] == 88.0

    chunks = TableSemanticChunker().chunk_dataset_profile(prod_job_result)
    joined = "\n".join(c["content"] for c in chunks)
    assert "sales_ecommerce" in joined
    assert "Missing region labels" in joined

    # Dict-shaped recommendations must not crash with TypeError on recs[:3] or recommend fallback
    rec_html = _generate_smart_dataset_answer("What are your recommendations?", prod_job_result)
    assert "Segment by Region" in rec_html
    overview_html = _generate_smart_dataset_answer("Tell me about this dataset", prod_job_result)
    assert "Segment by Region" in overview_html
    qual_html = _generate_smart_dataset_answer("Any quality issues?", prod_job_result)
    assert "Missing region labels" in qual_html
    assert "88.0%" in qual_html


# ─── BG-05, BG-06, BG-07, BG-08: Caches, Streams, Tickets, Steps, & Routes ───


def test_bg05_superset_cache_invalidation():
    from app.services.superset_engine import SupersetEngine

    engine = SupersetEngine()
    df_v1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    engine.prewarm_dataset("task-bi", "ref1", df_v1, file_path="/tmp/cleaned_task-bi.parquet")
    assert engine.resolve_dataset_df("task-bi").width == 2
    _ = engine.introspect_dataset("task-bi")
    assert "task-bi" in engine._schema_cache

    engine.invalidate_task("task-bi", "/tmp/cleaned_task-bi.parquet")
    assert "task-bi" not in engine._task_df_cache
    assert "task-bi" not in engine._schema_cache
    assert "/tmp/cleaned_task-bi.parquet" not in engine._df_cache


def test_bg06_stream_tickets_redis_and_fallback(monkeypatch):
    from app.core import auth

    fake_redis_store: dict[str, str] = {}

    class FakeRedis:
        def setex(self, key: str, ttl: int, val: str):
            fake_redis_store[key] = val

        def getdel(self, key: str):
            return fake_redis_store.pop(key, None)

    monkeypatch.setattr(auth, "_ticket_redis_client", FakeRedis())
    monkeypatch.setattr(auth, "_ticket_redis_checked", True)

    t1 = auth.create_stream_ticket(ttl_seconds=60)
    assert any(t1 in k for k in fake_redis_store)
    assert auth.consume_stream_ticket(t1) is True
    assert auth.consume_stream_ticket(t1) is False
    assert auth.consume_stream_ticket(auth.create_stream_ticket(ttl_seconds=0)) is False


def test_bg07_chat_stream_media_type_and_sse_framing(monkeypatch):
    from app.services.rag_service import rag_service
    from app.services.task_manager import TaskStatus, title_task_manager

    task_id = str(uuid.uuid4())
    fake_job = SimpleNamespace(
        task_id=task_id,
        owner_id=None,
        status=TaskStatus.COMPLETED,
        result={"filename": "sample.csv", "analysis": {"summary": {}}},
    )
    monkeypatch.setattr(title_task_manager, "get_job_async", AsyncMock(return_value=fake_job))

    async def fake_stream(*args, **kwargs):
        yield json.dumps({"type": "token", "token": "Hello"}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    monkeypatch.setattr(rag_service, "chat_stream_with_report", fake_stream)
    client = TestClient(app)

    ndjson_resp = client.post(f"/api/jobs/{task_id}/chat/stream", json={"question": "Summary"})
    assert ndjson_resp.status_code == 200
    assert ndjson_resp.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(l) for l in ndjson_resp.text.strip().splitlines()]
    assert lines[0]["token"] == "Hello"

    sse_resp = client.post(
        f"/api/jobs/{task_id}/chat/stream",
        json={"question": "Summary"},
        headers={"Accept": "text/event-stream"},
    )
    assert sse_resp.status_code == 200
    assert sse_resp.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"type": "token", "token": "Hello"}\n\n' in sse_resp.text


def test_bg08_pipeline_pii_dq_steps_hiplot_route_and_join_fk(monkeypatch):
    from app.services.analysis_pipeline import run_pipeline
    from app.services.superset_engine import superset_engine

    df_pii = pl.DataFrame({
        "email": ["alice@example.com", "bob@example.com", "carol@example.com"],
        "amount": [10.0, 20.0, 30.0],
    })
    pipeline_res: dict = {"summary": {"amount": {"mean": 20.0, "min": 10.0, "max": 30.0}}}
    run_pipeline(df_pii, pipeline_res)
    assert pipeline_res.get("pii", {}).get("has_pii") is True
    assert "email" in pipeline_res["pii"]["findings"]
    assert pipeline_res.get("data_quality", {}).get("total_tests", 0) > 0

    task_id = str(uuid.uuid4())
    monkeypatch.setattr(superset_engine, "resolve_dataset_df", lambda tid: df_pii)
    client = TestClient(app)
    hi_resp = client.get(f"/api/analytics/explore/{task_id}/hiplot")
    assert hi_resp.status_code == 200
    hi_data = hi_resp.json()
    assert hi_data["total_rows"] == 3
    assert len(hi_data["dimensions"]) == 2

    # /upload/join referential integrity notice on orphaned FK rows
    from app.api.routes import upload as upload_route
    monkeypatch.setattr(upload_route, "dispatch_task", AsyncMock(return_value=None))
    from app.services.task_manager import title_task_manager
    monkeypatch.setattr(title_task_manager, "create_job_async", AsyncMock(return_value=str(uuid.uuid4())))
    monkeypatch.setattr(title_task_manager, "update_result_async", AsyncMock(return_value=None))

    parent_csv = BytesIO(b"user_id,name\n1,Alice\n2,Bob\n")
    child_csv = BytesIO(b"user_id,spend\n1,50\n99,120\n")
    join_resp = client.post(
        "/api/upload/join",
        files=[
            ("files", ("users.csv", parent_csv, "text/csv")),
            ("files", ("orders.csv", child_csv, "text/csv")),
        ],
        data={"join_key": "user_id", "join_type": "left"},
    )
    assert join_resp.status_code == 200
    assert "Referential integrity notice: 1 unmatched foreign-key row(s)" in join_resp.json()["message"]
