import pytest
import numpy as np
from datetime import datetime, timedelta
import polars as pl
from app.services.timeseries_kats import (
    CUSUMChangepointDetector,
    HoltWintersForecaster,
    TSFeaturesExtractor,
    run_kats_time_series_intelligence
)
from app.services.analysis.time_series import analyze_time_series


def test_cusum_changepoint_detection():
    # Step change: first 50 points around mean 10, next 50 points around mean 40
    np.random.seed(42)
    s1 = np.random.normal(loc=10.0, scale=1.0, size=50)
    s2 = np.random.normal(loc=40.0, scale=1.0, size=50)
    series = np.concatenate([s1, s2])

    timestamps = [f"2026-01-{i+1:02d}" for i in range(31)] + [f"2026-02-{i+1:02d}" for i in range(28)] + [f"2026-03-{i+1:02d}" for i in range(31)] + [f"2026-04-{i+1:02d}" for i in range(10)]

    detector = CUSUMChangepointDetector(significance_level=0.05, min_distance=5)
    changepoints = detector.detect_changepoints(series, timestamps=timestamps)

    assert len(changepoints) >= 1
    primary_cp = changepoints[0]
    # Should detect changepoint close to index 50
    assert 45 <= primary_cp["index"] <= 55
    assert primary_cp["statistically_significant"] is True
    assert primary_cp["pre_mean"] < 15.0
    assert primary_cp["post_mean"] > 35.0
    assert primary_cp["mean_shift"] > 20.0


def test_holt_winters_forecasting():
    # Linear upward trend with slight noise
    x = np.arange(60)
    series = 5.0 + 0.8 * x + np.random.normal(0, 0.5, size=60)

    forecaster = HoltWintersForecaster(alpha=0.3, beta=0.2)
    res = forecaster.forecast(series, horizon=20)

    assert res["success"] is True
    assert res["horizon"] == 20
    assert len(res["forecast"]) == 20
    assert len(res["lower_bound_95"]) == 20
    assert len(res["upper_bound_95"]) == 20

    # Trend should continue upward
    assert res["forecast"][-1] > res["forecast"][0]
    # Check 95% confidence bounds
    for i in range(20):
        assert res["upper_bound_95"][i] >= res["forecast"][i] >= res["lower_bound_95"][i]


def test_tsfeatures_extraction():
    # 1. Trending series
    trending = np.linspace(10, 100, 80) + np.random.normal(0, 1, size=80)
    feats = TSFeaturesExtractor.extract_features(trending)

    assert feats["valid"] is True
    assert feats["trend_strength"] > 0.7
    assert feats["classification"] == "strongly_trending"

    # 2. Constant series
    constant = np.ones(50) * 42.0
    const_feats = TSFeaturesExtractor.extract_features(constant)
    assert const_feats["valid"] is True
    assert const_feats["classification"] == "constant"


def test_run_kats_time_series_intelligence_dataframe():
    base_date = datetime(2026, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(70)]
    # Series with a clear changepoint at day 35
    vals = [20.0 + np.random.normal(0, 1) for _ in range(35)] + [80.0 + np.random.normal(0, 1) for _ in range(35)]

    df = pl.DataFrame({
        "timestamp": dates,
        "revenue": vals
    })

    res = run_kats_time_series_intelligence(df, "timestamp", "revenue", forecast_horizon=15)
    assert res["success"] is True
    assert res["has_structural_changes"] is True
    assert len(res["changepoints"]) >= 1
    assert "forecast" in res
    assert len(res["forecast"]["forecast"]) == 15

    # Test integration inside analyze_time_series
    full_analysis = analyze_time_series(df, ["revenue"])
    assert full_analysis["has_time_series"] is True
    assert "kats_intelligence" in full_analysis
    assert "revenue" in full_analysis["kats_intelligence"]
    assert len(full_analysis["changepoints"]["revenue"]) >= 1
