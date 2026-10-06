"""
Optimal Clustering Engine
=========================
Determines optimal cluster count K and partitions records using
K-Means and MiniBatchKMeans with Silhouette Score evaluation.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class OptimalKMeansClusterer:
    """
    Automated cluster count discovery and partitioning engine.
    """

    @classmethod
    def fit_optimal_clusters(
        cls,
        X: np.ndarray,
        min_k: int = 2,
        max_k: int = 6,
        random_state: int = 42,
    ) -> Tuple[int, float, np.ndarray]:
        """
        Searches K in [min_k, max_k] using silhouette score.

        Returns:
            Tuple of (optimal_k, silhouette_score, cluster_labels)
        """
        n_samples = X.shape[0]
        actual_max_k = min(max_k, max(min_k, n_samples // 10))

        if actual_max_k <= min_k:
            actual_max_k = min_k

        # Subsample for silhouette score evaluation if dataset is large
        if n_samples > 3000:
            rng = np.random.default_rng(random_state)
            sample_idx = rng.choice(n_samples, 3000, replace=False)
            X_eval = X[sample_idx]
        else:
            X_eval = X

        best_k = min_k
        best_score = -1.0
        best_labels: Optional[np.ndarray] = None

        use_minibatch = n_samples > 5000

        for k in range(min_k, actual_max_k + 1):
            if use_minibatch:
                model = MiniBatchKMeans(
                    n_clusters=k,
                    batch_size=min(1024, n_samples),
                    random_state=random_state,
                    n_init="auto",
                )
            else:
                model = KMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init="auto",
                )

            labels = model.fit_predict(X)

            # Check if all clusters received membership
            unique_labels = np.unique(labels)
            if len(unique_labels) < k:
                continue

            if n_samples > 3000:
                eval_labels = labels[sample_idx]
            else:
                eval_labels = labels

            try:
                score = float(silhouette_score(X_eval, eval_labels))
            except Exception:
                score = 0.0

            # Adjusted score with slight parsimony bias against overly fragmented K
            adj_score = score - (0.015 * (k - min_k))
            if adj_score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        if best_labels is None:
            model = KMeans(n_clusters=min_k, random_state=random_state, n_init="auto")
            best_labels = model.fit_predict(X)
            best_k = min_k
            best_score = 0.0

        return best_k, max(0.0, best_score), best_labels
