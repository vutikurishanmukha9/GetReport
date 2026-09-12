"""
Non-linear, Non-monotonic, and Mixed-Type Correlation Engine.
Adapted from fg-data-profiling (Phik / phi_k framework).

Unlike Pearson (linear only) and Spearman (monotonic only), Phik (phi_k)
measures uniform, quadratic, cyclical, and categorical associations
between any pair of variables (numeric-numeric, numeric-categorical, categorical-categorical).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

PHIK_STRONG_THRESHOLD = 0.50
PHIK_VERY_STRONG_THRESHOLD = 0.75
MAX_ANALYSIS_COLUMNS = 16


def _discretize_series(
    series: pl.Series,
    num_bins: int = 10,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Discretize a Polars series into non-negative integer bins [0, n_bins - 1].
    Returns (binned_array, n_bins).
    """
    n = series.len()
    if n == 0:
        return None, 0

    dtype = series.dtype
    is_numeric = dtype.is_numeric()

    if is_numeric:
        valid_series = series.drop_nulls().cast(pl.Float64)
        if valid_series.len() < 5:
            return None, 0

        unique_count = valid_series.n_unique()
        if unique_count <= 1:
            return None, 0

        # If few distinct values, label encode directly
        if unique_count <= num_bins:
            unique_vals = sorted(valid_series.unique().to_list())
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}
            # Map full series (keeping nulls as -1)
            raw_vals = series.cast(pl.Float64).to_list()
            binned = np.array([val_to_idx.get(v, -1) if v is not None else -1 for v in raw_vals], dtype=np.int32)
            return binned, len(unique_vals)

        # Quantile binning for continuous distributions
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        try:
            edges = np.unique(np.quantile(valid_series.to_numpy(), quantiles))
            if len(edges) < 3:
                # Fallback to linear spacing
                edges = np.linspace(valid_series.min(), valid_series.max(), num_bins + 1)
        except Exception:
            edges = np.linspace(float(valid_series.min()), float(valid_series.max()), num_bins + 1)

        raw_np = series.cast(pl.Float64).to_numpy()
        # np.digitize returns 1-based bin index
        digitized = np.digitize(raw_np, edges[1:-1])
        # Mask nulls as -1
        null_mask = series.is_null().to_numpy()
        digitized[null_mask] = -1
        n_actual_bins = len(edges) - 1
        return digitized, n_actual_bins

    elif dtype == pl.Boolean:
        raw_bools = series.to_list()
        binned = np.array([1 if v is True else (0 if v is False else -1) for v in raw_bools], dtype=np.int32)
        return binned, 2

    else:
        # Categorical / Utf8 string
        valid_series = series.drop_nulls()
        if valid_series.len() < 5:
            return None, 0

        counts = valid_series.value_counts(sort=True)
        n_unique = counts.height
        if n_unique <= 1:
            return None, 0

        # Take top (num_bins - 1) categories, map rest to "other" bucket
        top_cats = [str(r[0]) for r in counts.head(num_bins - 1).iter_rows() if r[0] is not None]
        cat_to_idx = {cat: i for i, cat in enumerate(top_cats)}
        other_idx = len(top_cats)

        raw_list = series.to_list()
        binned = []
        for v in raw_list:
            if v is None:
                binned.append(-1)
            else:
                s_val = str(v)
                binned.append(cat_to_idx.get(s_val, other_idx))

        n_bins = len(top_cats) + 1 if n_unique > len(top_cats) else len(top_cats)
        return np.array(binned, dtype=np.int32), n_bins


def compute_bivariate_phik(
    x_binned: np.ndarray,
    n_bins_x: int,
    y_binned: np.ndarray,
    n_bins_y: int,
) -> float:
    """
    Computes bias-corrected contingency association phi_k between two pre-binned vectors.
    """
    # Filter out null positions (marked as -1)
    valid_mask = (x_binned >= 0) & (y_binned >= 0)
    x_valid = x_binned[valid_mask]
    y_valid = y_binned[valid_mask]

    n_samples = len(x_valid)
    if n_samples < 10 or n_bins_x < 2 or n_bins_y < 2:
        return 0.0

    # Build 2D contingency table
    try:
        table, _, _ = np.histogram2d(
            x_valid,
            y_valid,
            bins=[n_bins_x, n_bins_y],
            range=[[0, n_bins_x], [0, n_bins_y]],
        )
    except Exception:
        return 0.0

    # Row and column marginals
    row_sums = np.sum(table, axis=1, keepdims=True)
    col_sums = np.sum(table, axis=0, keepdims=True)

    expected = (row_sums @ col_sums) / float(n_samples)

    nonzero_expected = expected > 1e-9
    if not np.any(nonzero_expected):
        return 0.0

    chi2 = np.sum((table[nonzero_expected] - expected[nonzero_expected]) ** 2 / expected[nonzero_expected])
    phi2 = chi2 / float(n_samples)

    r, c = table.shape
    if n_samples <= 1 or r <= 1 or c <= 1:
        return 0.0

    # Bergsma / Cramér bias correction
    phi2_corr = max(0.0, phi2 - ((r - 1.0) * (c - 1.0)) / (n_samples - 1.0))
    r_corr = r - ((r - 1.0) ** 2) / (n_samples - 1.0)
    c_corr = c - ((c - 1.0) ** 2) / (n_samples - 1.0)
    k = min(r_corr - 1.0, c_corr - 1.0)

    if k <= 0.0:
        return 0.0

    v = np.sqrt(phi2_corr / k)
    return float(np.clip(v, 0.0, 1.0))


def compute_phik_matrix(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
    num_bins: int = 10,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """
    Computes a complete, symmetric Phik (phi_k) correlation matrix across numeric,
    categorical, and boolean columns.

    Args:
        df: Input Polars DataFrame
        columns: Optional list of columns to include. If None, auto-selects up to 16
                 analytical numeric and non-high-cardinality categorical columns.
        num_bins: Discretization resolution for continuous variables.

    Returns:
        (phik_matrix, strong_associations)
        where phik_matrix is {col_a: {col_b: score}}
        and strong_associations is a sorted list of pairs with score >= 0.50.
    """
    if df.height < 5 or df.width < 2:
        return {}, []

    # Select target columns
    if columns is not None:
        target_cols = [c for c in columns if c in df.columns]
    else:
        # Auto-select suitable columns
        candidate_cols = []
        for c in df.columns:
            dtype = df[c].dtype
            if dtype.is_numeric() or dtype == pl.Boolean:
                # Exclude constant columns
                if df[c].drop_nulls().n_unique() > 1:
                    candidate_cols.append(c)
            elif dtype == pl.Utf8:
                # Exclude high-cardinality ID / free-text columns
                n_uniq = df[c].n_unique()
                if 2 <= n_uniq <= 50:
                    candidate_cols.append(c)

        target_cols = candidate_cols[:MAX_ANALYSIS_COLUMNS]

    if len(target_cols) < 2:
        return {}, []

    # Pre-discretize selected columns
    discretized: Dict[str, Tuple[Optional[np.ndarray], int]] = {}
    valid_cols: List[str] = []

    for col in target_cols:
        binned_arr, n_bins = _discretize_series(df[col], num_bins=num_bins)
        if binned_arr is not None and n_bins >= 2:
            discretized[col] = (binned_arr, n_bins)
            valid_cols.append(col)

    if len(valid_cols) < 2:
        return {}, []

    phik_matrix: Dict[str, Dict[str, float]] = {c: {} for c in valid_cols}
    strong_associations: List[Dict[str, Any]] = []

    for i, col_a in enumerate(valid_cols):
        phik_matrix[col_a][col_a] = 1.0
        arr_a, bins_a = discretized[col_a]

        for j in range(i + 1, len(valid_cols)):
            col_b = valid_cols[j]
            arr_b, bins_b = discretized[col_b]

            score = compute_bivariate_phik(arr_a, bins_a, arr_b, bins_b)
            score_rounded = round(score, 4)

            phik_matrix[col_a][col_b] = score_rounded
            phik_matrix[col_b][col_a] = score_rounded

            if score_rounded >= PHIK_STRONG_THRESHOLD:
                # Classify relationship types
                type_a = "numeric" if df[col_a].dtype.is_numeric() else "categorical"
                type_b = "numeric" if df[col_b].dtype.is_numeric() else "categorical"
                pair_type = f"{type_a}-{type_b}" if type_a == type_b else "mixed"

                strength = "very strong" if score_rounded >= PHIK_VERY_STRONG_THRESHOLD else "strong"

                strong_associations.append({
                    "column_a": col_a,
                    "column_b": col_b,
                    "phik": score_rounded,
                    "strength": strength,
                    "type": pair_type,
                })

    # Sort strong associations by phik descending
    strong_associations.sort(key=lambda item: item["phik"], reverse=True)

    return phik_matrix, strong_associations
