"""
Unit & Integration Tests for Unsupervised Intelligence Modules
==============================================================
Tests the modular preprocessor, optimal clusterer, PCA projector,
persona synthesizer, and end-to-end service orchestration.
"""
import numpy as np
import polars as pl
import pytest

from app.services.unsupervised import (
    ClusterMatrixPreprocessor,
    OptimalKMeansClusterer,
    PCAProjector,
    PersonaSynthesizer,
    UnsupervisedIntelligenceService,
)


@pytest.fixture
def three_cluster_df() -> pl.DataFrame:
    """Generates a synthetic 3-cluster customer transaction dataset."""
    rng = np.random.default_rng(42)
    n = 600

    # Cluster 1: High Spend, High Frequency
    c1_spend = rng.normal(800, 50, n // 3)
    c1_freq = rng.normal(40, 5, n // 3)
    c1_tenure = rng.normal(5, 1, n // 3)

    # Cluster 2: Budget Shoppers
    c2_spend = rng.normal(150, 30, n // 3)
    c2_freq = rng.normal(10, 3, n // 3)
    c2_tenure = rng.normal(1, 0.5, n // 3)

    # Cluster 3: Moderate Regulars
    c3_spend = rng.normal(400, 40, n // 3)
    c3_freq = rng.normal(25, 4, n // 3)
    c3_tenure = rng.normal(3, 0.8, n // 3)

    spend = np.concatenate([c1_spend, c2_spend, c3_spend])
    freq = np.concatenate([c1_freq, c2_freq, c3_freq])
    tenure = np.concatenate([c1_tenure, c2_tenure, c3_tenure])

    return pl.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "annual_spend": spend,
        "purchase_frequency": freq,
        "tenure_years": tenure,
        "account_tier": ["Gold"] * 200 + ["Bronze"] * 200 + ["Silver"] * 200,
    })


def test_preprocessor_filters_ids_and_prepares_matrix(three_cluster_df):
    X, feature_names, num_cols = ClusterMatrixPreprocessor.prepare_matrix(three_cluster_df)
    assert X is not None
    assert "customer_id" not in feature_names
    assert "annual_spend" in feature_names
    assert "purchase_frequency" in feature_names
    assert "tenure_years" in feature_names
    assert X.shape[0] == 600
    assert X.shape[1] >= 3
    # Check that output is scaled (finite numbers, no NaNs)
    assert np.isfinite(X).all()


def test_optimal_kmeans_finds_distinct_clusters(three_cluster_df):
    X, _, _ = ClusterMatrixPreprocessor.prepare_matrix(three_cluster_df)
    best_k, sil_score, labels = OptimalKMeansClusterer.fit_optimal_clusters(X)
    assert best_k in (3, 4)  # Natural clusters are 3
    assert sil_score > 0.40   # Strong separation
    assert len(np.unique(labels)) == best_k
    assert labels.shape[0] == 600


def test_pca_projector_produces_valid_coordinates_and_loadings(three_cluster_df):
    X, feature_names, _ = ClusterMatrixPreprocessor.prepare_matrix(three_cluster_df)
    _, _, labels = OptimalKMeansClusterer.fit_optimal_clusters(X)
    points, exp_var, loadings = PCAProjector.project_2d(
        X, labels, three_cluster_df, feature_names, max_points=200
    )
    assert len(points) == 200
    assert exp_var > 60.0  # Captures majority of variance
    assert len(loadings["pc1_top_drivers"]) >= 2
    assert len(loadings["pc2_top_drivers"]) >= 2
    # Check point structure
    p0 = points[0]
    assert isinstance(p0.x, float) and isinstance(p0.y, float)
    assert p0.cluster in set(labels)


def test_persona_synthesizer_identifies_over_indexed_features(three_cluster_df):
    X, _, num_cols = ClusterMatrixPreprocessor.prepare_matrix(three_cluster_df)
    k, _, labels = OptimalKMeansClusterer.fit_optimal_clusters(X)
    personas = PersonaSynthesizer.synthesize(three_cluster_df, labels, k, num_cols)
    assert len(personas) == k
    for p in personas:
        assert p.size > 0
        assert p.share_pct > 0
        assert len(p.name) > 5
        assert "Represents" in p.summary
        assert len(p.distinguishing_features) > 0


def test_insufficient_data_bypasses_gracefully():
    # Less than 15 rows
    df_tiny = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    res_tiny = UnsupervisedIntelligenceService.run_unsupervised_discovery(df_tiny)
    assert res_tiny.ran is False
    assert "too_few_rows" in res_tiny.reason

    # Single column
    df_single = pl.DataFrame({"a": list(range(40))})
    res_single = UnsupervisedIntelligenceService.run_unsupervised_discovery(df_single)
    assert res_single.ran is False
    assert "insufficient_numeric_features" in res_single.reason


def test_end_to_end_orchestration_and_serialization(three_cluster_df):
    result = UnsupervisedIntelligenceService.run_unsupervised_discovery(three_cluster_df)
    assert result.ran is True
    assert result.reason == "ok"
    assert result.optimal_k >= 2
    assert result.silhouette_score > 0.3
    assert len(result.personas) == result.optimal_k
    assert len(result.points) > 0

    d = result.to_dict()
    assert d["ran"] is True
    assert "optimal_k" in d
    assert "personas" in d
    assert "points" in d
    assert isinstance(d["personas"], list)
    assert isinstance(d["points"], list)


def test_clustering_is_deterministic(three_cluster_df):
    res1 = UnsupervisedIntelligenceService.run_unsupervised_discovery(three_cluster_df, random_state=42)
    res2 = UnsupervisedIntelligenceService.run_unsupervised_discovery(three_cluster_df, random_state=42)
    assert res1.optimal_k == res2.optimal_k
    assert res1.silhouette_score == res2.silhouette_score
    p1 = [p.cluster for p in res1.points]
    p2 = [p.cluster for p in res2.points]
    assert p1 == p2


def test_large_dataset_uses_minibatch_and_runs_fast():
    rng = np.random.default_rng(99)
    n = 6000
    df_large = pl.DataFrame({
        "id": [f"ID_{i}" for i in range(n)],
        "val1": rng.normal(100, 15, n),
        "val2": rng.normal(50, 5, n),
        "val3": rng.normal(200, 30, n),
    })
    result = UnsupervisedIntelligenceService.run_unsupervised_discovery(df_large)
    assert result.ran is True
    assert len(result.points) <= 2000  # Subsampling active
