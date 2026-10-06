"""
Unsupervised Learning Domain Schemas
====================================
Dataclasses and serialization models for clustering, PCA projection,
and contrastive persona synthesis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClusterPoint:
    """A projected 2D coordinate for interactive canvas/scatter visualization."""
    id: int
    x: float
    y: float
    cluster: int
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "cluster": self.cluster,
            "label": self.label,
        }


@dataclass
class ClusterFeatureDiff:
    """Over/under-indexed feature contrast vs the population baseline."""
    feature: str
    cluster_mean: float
    population_mean: float
    index_ratio: float
    direction: str  # "higher" | "lower"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "cluster_mean": round(self.cluster_mean, 2),
            "population_mean": round(self.population_mean, 2),
            "index_ratio": round(self.index_ratio, 2),
            "direction": self.direction,
        }


@dataclass
class ClusterPersona:
    """Synthesized business segment profile with distinguishing characteristics."""
    cluster_id: int
    name: str
    size: int
    share_pct: float
    distinguishing_features: List[ClusterFeatureDiff]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "size": self.size,
            "share_pct": round(self.share_pct, 1),
            "distinguishing_features": [f.to_dict() for f in self.distinguishing_features],
            "summary": self.summary,
        }


@dataclass
class ClusteringResult:
    """Complete result container for unsupervised discovery."""
    ran: bool
    reason: str
    optimal_k: int = 0
    silhouette_score: float = 0.0
    explained_variance_pct: float = 0.0
    pc_loadings: Dict[str, List[str]] = field(
        default_factory=lambda: {"pc1_top_drivers": [], "pc2_top_drivers": []}
    )
    personas: List[ClusterPersona] = field(default_factory=list)
    points: List[ClusterPoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ran": self.ran,
            "reason": self.reason,
            "optimal_k": self.optimal_k,
            "silhouette_score": round(self.silhouette_score, 3),
            "explained_variance_pct": round(self.explained_variance_pct, 1),
            "pc_loadings": self.pc_loadings,
            "personas": [p.to_dict() for p in self.personas],
            "points": [pt.to_dict() for pt in self.points],
        }
