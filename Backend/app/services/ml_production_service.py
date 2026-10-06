"""
ML Production Service Facade Module
Re-exports the modular ML production package components for backward compatibility.
"""

from app.services.ml_production import (
    ExecutiveMLSection,
    ModelArtifactMetadata,
    BatchPredictionResult,
    ExecutiveMLReportSynthesizer,
    ModelArtifactSerializer,
    BatchInferenceEngine,
    MLProductionService,
)

__all__ = [
    "ExecutiveMLSection",
    "ModelArtifactMetadata",
    "BatchPredictionResult",
    "ExecutiveMLReportSynthesizer",
    "ModelArtifactSerializer",
    "BatchInferenceEngine",
    "MLProductionService",
]
