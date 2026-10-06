"""
PCA 2D Dimensionality Projection
================================
Projects multi-dimensional feature matrices into 2D coordinates
for interactive canvas and scatter plot visual discovery.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from .schemas import ClusterPoint


def _tokens(name: str) -> Set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


class PCAProjector:
    """
    Computes PCA 2D coordinates, variance explained, and feature loadings.
    """

    @classmethod
    def project_2d(
        cls,
        X: np.ndarray,
        labels: np.ndarray,
        df: pl.DataFrame,
        feature_names: List[str],
        max_points: int = 2000,
        random_state: int = 42,
    ) -> Tuple[List[ClusterPoint], float, Dict[str, List[str]]]:
        """
        Projects X to 2D coordinates and extracts top feature loadings.

        Returns:
            Tuple of (points, explained_variance_pct, pc_loadings)
        """
        pca = PCA(n_components=2, random_state=random_state)
        coords_2d = pca.fit_transform(X)
        exp_var = float(np.sum(pca.explained_variance_ratio_) * 100.0)

        # Feature loadings
        loadings = pca.components_
        pc1_loadings = np.abs(loadings[0])
        pc2_loadings = np.abs(loadings[1])

        top_pc1_idx = np.argsort(-pc1_loadings)[:3]
        top_pc2_idx = np.argsort(-pc2_loadings)[:3]

        pc1_top = [feature_names[i] for i in top_pc1_idx if i < len(feature_names)]
        pc2_top = [feature_names[i] for i in top_pc2_idx if i < len(feature_names)]

        # Stratified sampling if data exceeds max_points
        n_samples = X.shape[0]
        if n_samples > max_points:
            rng = np.random.default_rng(random_state)
            indices: List[int] = []
            per_cluster = max_points // max(1, len(np.unique(labels)))
            for c in np.unique(labels):
                c_idx = np.flatnonzero(labels == c)
                if len(c_idx) > per_cluster:
                    indices.extend(rng.choice(c_idx, per_cluster, replace=False).tolist())
                else:
                    indices.extend(c_idx.tolist())
            if len(indices) < max_points:
                remaining = list(set(range(n_samples)) - set(indices))
                needed = max_points - len(indices)
                if remaining and needed > 0:
                    indices.extend(rng.choice(remaining, min(len(remaining), needed), replace=False).tolist())
            indices = sorted(indices)
        else:
            indices = list(range(n_samples))

        # Look for human-readable label candidate column
        label_col = next(
            (c for c in df.columns if _tokens(c) & {"name", "title", "customer", "product", "item"}),
            None,
        )

        points: List[ClusterPoint] = []
        for i in indices:
            row_label = str(df[label_col][i]) if label_col and df[label_col][i] is not None else None
            points.append(
                ClusterPoint(
                    id=int(i),
                    x=float(coords_2d[i, 0]),
                    y=float(coords_2d[i, 1]),
                    cluster=int(labels[i]),
                    label=row_label,
                )
            )

        return points, exp_var, {"pc1_top_drivers": pc1_top, "pc2_top_drivers": pc2_top}
