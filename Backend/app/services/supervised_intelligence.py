"""
Supervised Intelligence Facade Module
Re-exports the modular supervised AutoML package components for backward compatibility.
"""

from app.services.supervised import (
    TaskType,
    KeyDriver,
    LeaderboardEntry,
    AutoMLResult,
    TargetInferenceService,
    AutoMLMatrixPreprocessor,
    AutoMLModelTrainer,
    KeyDriverAnalyzer,
    SupervisedAutoMLService,
)

__all__ = [
    "TaskType",
    "KeyDriver",
    "LeaderboardEntry",
    "AutoMLResult",
    "TargetInferenceService",
    "AutoMLMatrixPreprocessor",
    "AutoMLModelTrainer",
    "KeyDriverAnalyzer",
    "SupervisedAutoMLService",
]
