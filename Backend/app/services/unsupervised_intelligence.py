"""
Backward-compatible facade forwarding to modular app.services.unsupervised package.
"""
from app.services.unsupervised import (
    ClusterFeatureDiff,
    ClusterPersona,
    ClusterPoint,
    ClusteringResult,
    UnsupervisedIntelligenceService,
    ClusterMatrixPreprocessor,
    OptimalKMeansClusterer,
    PCAProjector,
    PersonaSynthesizer,
)

__all__ = [
    "UnsupervisedIntelligenceService",
    "ClusteringResult",
    "ClusterPersona",
    "ClusterPoint",
    "ClusterFeatureDiff",
    "ClusterMatrixPreprocessor",
    "OptimalKMeansClusterer",
    "PCAProjector",
    "PersonaSynthesizer",
]
