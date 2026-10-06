"""
Unsupervised Feature Matrix Preprocessor
========================================
Extracts, encodes, imputes, and scales features for unsupervised models.
Excludes identifier keys, timestamps, and constant/near-zero variance columns.
"""
from __future__ import annotations

import logging
import math
import re
from typing import List, Optional, Set, Tuple

import numpy as np
import polars as pl
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

_ID_TOKENS = frozenset({
    "id", "uuid", "guid", "code", "sku", "zip", "zipcode", "pin", "phone",
    "mobile", "key", "index", "row_id", "__row_id__"
})


def _tokens(name: str) -> Set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def _is_id_column(name: str) -> bool:
    return bool(_tokens(name) & _ID_TOKENS)


class ClusterMatrixPreprocessor:
    """
    Prepares a clean, scaled NumPy feature matrix from a tabular Polars DataFrame.
    """

    @classmethod
    def prepare_matrix(
        cls,
        df: pl.DataFrame,
        max_features: int = 25,
    ) -> Tuple[Optional[np.ndarray], List[str], List[str]]:
        """
        Extracts robust scaled features from numeric and low-cardinality columns.

        Returns:
            (scaled_matrix, feature_names, original_numeric_cols)
            or (None, [], []) if data is insufficient for clustering.
        """
        n_rows = df.height
        if n_rows < 15 or df.width < 2:
            return None, [], []

        numeric_cols: List[str] = []
        low_card_cats: List[str] = []

        for col in df.columns:
            if _is_id_column(col):
                continue
            s = df[col]
            dtype = s.dtype

            if dtype.is_numeric():
                std_val = s.std()
                if std_val is not None and not math.isnan(std_val) and std_val > 1e-6:
                    numeric_cols.append(col)
            elif dtype in (pl.Utf8, pl.String, pl.Categorical):
                n_uniq = s.n_unique()
                if 2 <= n_uniq <= 12 and n_rows >= 30:
                    low_card_cats.append(col)

        # Cap features for optimal performance
        numeric_cols = numeric_cols[:max_features]
        remaining_slots = max(0, max_features - len(numeric_cols))
        low_card_cats = low_card_cats[:remaining_slots]

        if len(numeric_cols) < 2 and (len(numeric_cols) + len(low_card_cats)) < 2:
            return None, [], []

        feature_arrays: List[np.ndarray] = []
        feature_names: List[str] = []

        # 1. Process numeric features
        for col in numeric_cols:
            vals = df[col].cast(pl.Float32, strict=False).to_numpy()
            nan_mask = np.isnan(vals)
            if nan_mask.any():
                med = float(np.nanmedian(vals)) if not np.all(nan_mask) else 0.0
                vals[nan_mask] = med
            feature_arrays.append(vals.reshape(-1, 1))
            feature_names.append(col)

        # 2. Process low-cardinality categoricals via normalized frequency encoding
        for col in low_card_cats:
            counts = df[col].value_counts().to_dicts()
            freq_map = {row[col]: row["count"] / n_rows for row in counts if row[col] is not None}
            col_vals = df[col].to_list()
            freq_vec = np.array([freq_map.get(v, 0.0) for v in col_vals], dtype=np.float32).reshape(-1, 1)
            feature_arrays.append(freq_vec)
            feature_names.append(f"{col}_freq")

        if not feature_arrays:
            return None, [], []

        X = np.hstack(feature_arrays).astype(np.float32)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        return X_scaled, feature_names, numeric_cols
