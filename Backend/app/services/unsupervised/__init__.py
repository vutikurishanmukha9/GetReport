"""
Unsupervised Learning Package
=============================
Modular unsupervised machine learning for GetReport.
"""
from .engine import UnsupervisedIntelligenceService
from .schemas import (
    ClusterFeatureDiff,
    ClusterPersona,
    ClusterPoint,
    ClusteringResult,
)
from .preprocessor import ClusterMatrixPreprocessor
from .clustering import OptimalKMeansClusterer
from .projection import PCAProjector
from .persona_synthesizer import PersonaSynthesizer

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
