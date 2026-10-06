"""
Unsupervised Intelligence Engine
================================
Thin, non-monolithic orchestrator coordinating feature preprocessing,
optimal K clustering, PCA 2D projection, and persona synthesis.
"""
from __future__ import annotations

import logging
import time

import polars as pl

from .clustering import OptimalKMeansClusterer
from .persona_synthesizer import PersonaSynthesizer
from .preprocessor import ClusterMatrixPreprocessor
from .projection import PCAProjector
from .schemas import ClusteringResult

logger = logging.getLogger(__name__)


class UnsupervisedIntelligenceService:
    """
    Orchestrates decoupled unsupervised learning sub-modules.
    """

    @classmethod
    def run_unsupervised_discovery(
        cls,
        df: pl.DataFrame,
        max_sample_points: int = 2000,
        random_state: int = 42,
    ) -> ClusteringResult:
        """
        Executes end-to-end unsupervised clustering, PCA projection, and persona synthesis.
        """
        if df.is_empty():
            return ClusteringResult(ran=False, reason="empty_dataframe")

        if df.height < 15:
            return ClusteringResult(ran=False, reason="too_few_rows (minimum 15 rows required)")

        # Subsample large datasets for memory and latency efficiency
        sample_df = df.sample(n=1500, seed=random_state) if df.height > 1500 else df

        start_time = time.perf_counter()

        try:
            # 1. Modular matrix preprocessing
            X, feature_names, numeric_cols = ClusterMatrixPreprocessor.prepare_matrix(sample_df)
            if X is None or X.shape[1] < 2:
                return ClusteringResult(
                    ran=False,
                    reason="insufficient_numeric_features (minimum 2 valid numeric features required)",
                )

            # 2. Optimal K clustering
            optimal_k, sil_score, labels = OptimalKMeansClusterer.fit_optimal_clusters(
                X, random_state=random_state
            )

            # 3. PCA 2D projection
            points, exp_var, loadings = PCAProjector.project_2d(
                X,
                labels,
                sample_df,
                feature_names,
                max_points=max_sample_points,
                random_state=random_state,
            )

            # 4. Persona & contrast synthesis
            personas = PersonaSynthesizer.synthesize(sample_df, labels, optimal_k, numeric_cols)

            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Unsupervised discovery completed in %.1fms: k=%d, silhouette=%.3f, explained_var=%.1f%%",
                duration_ms, optimal_k, sil_score, exp_var,
            )

            return ClusteringResult(
                ran=True,
                reason="ok",
                optimal_k=optimal_k,
                silhouette_score=sil_score,
                explained_variance_pct=exp_var,
                pc_loadings=loadings,
                personas=personas,
                points=points,
            )

        except Exception as exc:
            logger.exception("Unsupervised discovery error: %s", exc)
            return ClusteringResult(ran=False, reason=f"execution_error: {str(exc)}")
