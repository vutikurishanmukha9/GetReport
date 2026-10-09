from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ExecutiveMLSection:
    has_ml_content: bool
    title: str = "Key Performance Drivers & Operational Cohorts"
    executive_takeaways: List[str] = field(default_factory=list)
    segmentation_summary: Optional[Dict[str, Any]] = None
    driver_attribution: Optional[Dict[str, Any]] = None
    model_governance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_ml_content": self.has_ml_content,
            "title": self.title,
            "executive_takeaways": self.executive_takeaways,
            "segmentation_summary": self.segmentation_summary,
            "driver_attribution": self.driver_attribution,
            "model_governance": self.model_governance,
        }

@dataclass
class ModelArtifactMetadata:
    model_name: str
    task_type: str
    target_column: str
    feature_names: List[str]
    performance_metrics: Dict[str, float]
    created_at_iso: str
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "target_column": self.target_column,
            "feature_names": self.feature_names,
            "performance_metrics": {k: round(v, 4) for k, v in self.performance_metrics.items()},
            "created_at_iso": self.created_at_iso,
            "version": self.version,
        }

@dataclass
class BatchPredictionResult:
    success: bool
    total_rows_scored: int
    predictions_col: str
    probabilities_col: Optional[str] = None
    cluster_col: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "total_rows_scored": self.total_rows_scored,
            "predictions_col": self.predictions_col,
            "probabilities_col": self.probabilities_col,
            "cluster_col": self.cluster_col,
            "error_message": self.error_message,
        }
