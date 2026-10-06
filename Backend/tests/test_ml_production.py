import os
import tempfile
import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from app.services.ml_production import (
    ExecutiveMLSection,
    ModelArtifactMetadata,
    ExecutiveMLReportSynthesizer,
    ModelArtifactSerializer,
    BatchInferenceEngine,
    MLProductionService,
)
from app.services.report_generator import generate_pdf_report

def test_report_synthesizer_empty():
    synthesizer = ExecutiveMLReportSynthesizer()
    section = synthesizer.synthesize_section(
        unsupervised_data={"ran": False},
        supervised_data={"ran": False}
    )
    assert section.has_ml_content is False
    assert len(section.executive_takeaways) == 0

def test_report_synthesizer_full():
    synthesizer = ExecutiveMLReportSynthesizer()
    unsupervised_data = {
        "ran": True,
        "optimal_k": 3,
        "silhouette_score": 0.582,
        "explained_variance_pct": 74.2,
        "personas": [
            {
                "cluster_id": 0,
                "name": "Cluster 1: High Spend, Low Churn",
                "size": 120,
                "share_pct": 45.0,
                "distinguishing_features": [
                    {"feature": "monthly_charges", "direction": "higher", "index_ratio": 1.45}
                ],
                "summary": "High value account segment"
            }
        ]
    }
    supervised_data = {
        "ran": True,
        "target_column": "churn",
        "task_type": "binary_classification",
        "best_model_name": "HistGradientBoostingClassifier",
        "primary_metric_name": "accuracy",
        "primary_metric_value": 0.895,
        "key_drivers": [
            {
                "feature": "tenure",
                "importance_pct": 34.5,
                "direction": "negative",
                "impact_description": "Higher tenure reduces churn probability"
            }
        ],
        "leaderboard": [
            {"model_name": "HistGradientBoostingClassifier", "primary_metric_value": 0.895, "train_time_sec": 0.12}
        ],
        "metrics": {"accuracy": 0.895, "f1_score": 0.870},
        "total_execution_time_sec": 0.45
    }

    section = synthesizer.synthesize_section(unsupervised_data, supervised_data)
    assert section.has_ml_content is True
    assert len(section.executive_takeaways) >= 2
    assert section.segmentation_summary is not None
    assert section.segmentation_summary["optimal_k"] == 3
    assert section.driver_attribution is not None
    assert section.driver_attribution["target_column"] == "churn"
    assert section.model_governance is not None

def test_model_artifact_serializer_roundtrip():
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    meta = ModelArtifactMetadata(
        model_name="HistGradientBoostingClassifier",
        task_type="binary_classification",
        target_column="churn",
        feature_names=["f1", "f2", "f3"],
        performance_metrics={"accuracy": 0.92},
        created_at_iso="2026-10-06T12:00:00Z"
    )

    serializer = ModelArtifactSerializer()
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = os.path.join(tmpdir, "model_bundle.joblib")
        saved_path = serializer.save_artifact(model, meta, artifact_path)
        assert os.path.exists(saved_path)

        loaded_model, loaded_meta, preproc = serializer.load_artifact(saved_path)
        assert loaded_meta.model_name == "HistGradientBoostingClassifier"
        assert loaded_meta.feature_names == ["f1", "f2", "f3"]
        assert loaded_meta.performance_metrics["accuracy"] == 0.92

        # Verify predictions match exactly
        orig_preds = model.predict(X[:5])
        reloaded_preds = loaded_model.predict(X[:5])
        np.testing.assert_array_equal(orig_preds, reloaded_preds)

def test_batch_inference_engine_success():
    np.random.seed(42)
    X = np.random.randn(60, 2)
    y = (X[:, 0] > 0).astype(int)

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    meta = ModelArtifactMetadata(
        model_name="HistGradientBoostingClassifier",
        task_type="binary_classification",
        target_column="conversion",
        feature_names=["var_a", "var_b"],
        performance_metrics={"accuracy": 0.88},
        created_at_iso="2026-10-06T12:00:00Z"
    )

    new_df = pl.DataFrame({
        "var_a": [0.5, -1.2, 2.1, 0.0],
        "var_b": [1.1, 0.4, -0.8, -1.5],
        "other_id": ["ID1", "ID2", "ID3", "ID4"]
    })

    scorer = BatchInferenceEngine()
    scored_df, res = scorer.score_dataframe(model, meta, new_df, {})

    assert res.success is True
    assert res.total_rows_scored == 4
    assert res.predictions_col == "predicted_conversion"
    assert "predicted_conversion" in scored_df.columns
    assert "predicted_conversion_probability" in scored_df.columns
    assert "other_id" in scored_df.columns  # Preserves existing columns

def test_batch_inference_missing_feature_error():
    model = HistGradientBoostingClassifier(random_state=42)
    meta = ModelArtifactMetadata(
        model_name="HistGradientBoostingClassifier",
        task_type="binary_classification",
        target_column="conversion",
        feature_names=["required_col_1", "required_col_2"],
        performance_metrics={"accuracy": 0.88},
        created_at_iso="2026-10-06T12:00:00Z"
    )

    incomplete_df = pl.DataFrame({"wrong_col": [1, 2, 3]})
    scorer = BatchInferenceEngine()
    _, res = scorer.score_dataframe(model, meta, incomplete_df, {})

    assert res.success is False
    assert "Missing" in (res.error_message or "")

def test_pdf_report_enrichment_integration():
    analysis_results = {
        "dataset_name": "Test Cohort",
        "row_count": 100,
        "column_count": 5,
        "missing_cells": 0,
        "duplicate_rows": 0,
        "health_score": 95,
        "executive_summary": "Clean dataset.",
        "unsupervised_learning": {
            "ran": True,
            "optimal_k": 2,
            "silhouette_score": 0.65,
            "explained_variance_pct": 80.0,
            "personas": []
        },
        "supervised_learning": {
            "ran": True,
            "target_column": "target",
            "task_type": "binary_classification",
            "best_model_name": "HistGradientBoostingClassifier",
            "primary_metric_name": "accuracy",
            "primary_metric_value": 0.91,
            "key_drivers": [],
            "leaderboard": [],
            "metrics": {}
        }
    }
    charts = {}

    buf, meta = generate_pdf_report(analysis_results, charts, "test_output.pdf")
    assert buf.getbuffer().nbytes > 100
    assert "executive_ml" in analysis_results
    assert analysis_results["executive_ml"]["has_ml_content"] is True
