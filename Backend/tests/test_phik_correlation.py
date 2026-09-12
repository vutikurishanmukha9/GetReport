"""
Tests for Phik (phi_k) Non-linear & Mixed-Type Correlation Matrix.
Adapted from fg-data-profiling.
"""
import polars as pl
import numpy as np
from app.services.analysis.correlations import (
    compute_phik_matrix,
    _discretize_series,
    compute_bivariate_phik,
)
from app.services.analysis.core import analyze_dataset


def test_discretize_numeric_and_categorical():
    # Numeric continuous
    num_s = pl.Series("num", np.linspace(0, 100, 50))
    binned_num, n_bins_num = _discretize_series(num_s, num_bins=5)
    assert binned_num is not None
    assert n_bins_num >= 3
    assert len(binned_num) == 50

    # Categorical string
    cat_s = pl.Series("cat", ["Alpha", "Beta", "Gamma", "Alpha", "Beta"] * 10)
    binned_cat, n_bins_cat = _discretize_series(cat_s, num_bins=5)
    assert binned_cat is not None
    assert n_bins_cat == 3

    # Boolean
    bool_s = pl.Series("flag", [True, False, True, False] * 10)
    binned_bool, n_bins_bool = _discretize_series(bool_s)
    assert binned_bool is not None
    assert n_bins_bool == 2


def test_phik_detects_nonlinear_quadratic_relationship():
    n = 400
    x = np.linspace(-10, 10, n)
    y = x ** 2  # Parabola, symmetric around 0

    # Pearson correlation is near zero due to symmetry
    pearson_r = np.corrcoef(x, y)[0, 1]
    assert abs(pearson_r) < 0.10

    df = pl.DataFrame({"x": x, "y": y})
    matrix, strong = compute_phik_matrix(df, columns=["x", "y"])

    # Phik should capture the non-linear relationship
    assert matrix["x"]["y"] > 0.50
    assert matrix["y"]["x"] == matrix["x"]["y"]
    assert len(strong) == 1
    assert strong[0]["column_a"] in ("x", "y")
    assert strong[0]["column_b"] in ("x", "y")


def test_phik_mixed_types_and_noise_separation():
    n = 300
    np.random.seed(42)
    score = np.linspace(0, 100, n)
    tier = np.where(score > 70, "Elite", np.where(score > 40, "Standard", "Basic"))
    noise = np.random.randn(n)

    df = pl.DataFrame({
        "score": score,
        "tier": tier,
        "noise": noise,
    })

    matrix, strong = compute_phik_matrix(df)

    # Score and tier should have very strong mixed association
    assert matrix["score"]["tier"] >= 0.70
    assert matrix["tier"]["score"] == matrix["score"]["tier"]

    # Noise should have low association
    assert matrix["score"]["noise"] < 0.35
    assert matrix["tier"]["noise"] < 0.35

    # Check strong pairs
    pair_names = {(s["column_a"], s["column_b"]) for s in strong}
    assert ("score", "tier") in pair_names or ("tier", "score") in pair_names


def test_phik_matrix_symmetry_and_diagonal():
    df = pl.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
        "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 5,
        "c": ["X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z", "X"] * 5,
    })

    matrix, _ = compute_phik_matrix(df)

    for col in ("a", "b", "c"):
        assert matrix[col][col] == 1.0

    assert matrix["a"]["b"] == matrix["b"]["a"]
    assert matrix["a"]["c"] == matrix["c"]["a"]
    assert matrix["b"]["c"] == matrix["c"]["b"]


def test_analyze_dataset_contains_phik_results():
    df = pl.DataFrame({
        "revenue": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0] * 5,
        "orders": [1, 2, 3, 4, 5, 6] * 5,
        "segment": ["Enterprise", "SMB", "SMB", "Enterprise", "Consumer", "Consumer"] * 5,
    })

    analysis = analyze_dataset(df)

    assert "phik_matrix" in analysis
    assert "phik_strong_associations" in analysis
    assert isinstance(analysis["phik_matrix"], dict)
    assert isinstance(analysis["phik_strong_associations"], list)
