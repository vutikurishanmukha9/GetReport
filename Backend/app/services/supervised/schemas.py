from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"

@dataclass
class KeyDriver:
    feature: str
    importance_pct: float
    direction: str  # "positive", "negative", or "neutral"
    impact_description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "importance_pct": round(self.importance_pct, 2),
            "direction": self.direction,
            "impact_description": self.impact_description,
        }

@dataclass
class LeaderboardEntry:
    model_name: str
    primary_metric_name: str
    primary_metric_value: float
    all_metrics: Dict[str, float]
    train_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_value": round(self.primary_metric_value, 4),
            "all_metrics": {k: round(v, 4) for k, v in self.all_metrics.items()},
            "train_time_sec": round(self.train_time_sec, 4),
        }

@dataclass
class AutoMLResult:
    ran: bool = False
    reason: Optional[str] = None
    target_column: Optional[str] = None
    task_type: Optional[str] = None
    best_model_name: Optional[str] = None
    primary_metric_name: Optional[str] = None
    primary_metric_value: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    leaderboard: List[LeaderboardEntry] = field(default_factory=list)
    key_drivers: List[KeyDriver] = field(default_factory=list)
    summary: str = ""
    total_execution_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ran": bool(self.ran),
            "reason": self.reason,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "best_model_name": self.best_model_name,
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_value": round(self.primary_metric_value, 4) if self.primary_metric_value is not None else None,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "leaderboard": [entry.to_dict() for entry in self.leaderboard],
            "key_drivers": [driver.to_dict() for driver in self.key_drivers],
            "summary": self.summary,
            "total_execution_time_sec": round(self.total_execution_time_sec, 4),
        }
