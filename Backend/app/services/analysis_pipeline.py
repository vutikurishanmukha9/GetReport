"""
analysis_pipeline.py
~~~~~~~~~~~~~~~~~~~~
Pipeline pattern for extensible analysis steps.

Each step is a callable that takes (df, result_dict) and enriches
the result dict in-place. Steps declare a ``name`` for logging and a
narrow exception tuple so unexpected errors propagate immediately.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, Tuple, Type

import polars as pl

logger = logging.getLogger(__name__)

# Narrow exception types that indicate *data* problems, not *code* bugs.
# ImportError, AttributeError, SyntaxError, etc. will propagate and crash.
SAFE_DATA_ERRORS: Tuple[Type[Exception], ...] = (
    ValueError,
    TypeError,
    ArithmeticError,
    KeyError,
    IndexError,
    pl.exceptions.PolarsError,
)


class AnalysisStep(Protocol):
    """Protocol every pipeline step must satisfy."""

    name: str

    def __call__(self, df: pl.DataFrame, result: dict[str, Any]) -> None:
        """Enrich *result* in-place. May raise SAFE_DATA_ERRORS on bad data."""
        ...


@dataclass
class StepResult:
    """Outcome of running one pipeline step."""
    name: str
    success: bool
    error: str | None = None


@dataclass
class PipelineResult:
    """Aggregate outcome of the whole pipeline."""
    steps: list[StepResult] = field(default_factory=list)

    @property
    def failed(self) -> list[StepResult]:
        return [s for s in self.steps if not s.success]


# ─── Concrete Steps ─────────────────────────────────────────────────────────

class ConfidenceStep:
    name = "confidence_scores"

    def __call__(self, df: pl.DataFrame, result: dict[str, Any]) -> None:
        from app.services.confidence_scoring import calculate_confidence_scores
        report = calculate_confidence_scores(df)
        result["confidence_scores"] = report.to_dict()
        logger.info(
            "Confidence scoring: Dataset grade=%s, %d high/%d low confidence columns",
            report._get_dataset_grade(),
            report.high_confidence_count,
            report.low_confidence_count,
        )


class DecisionsStep:
    name = "analysis_decisions"

    def __call__(self, df: pl.DataFrame, result: dict[str, Any]) -> None:
        from app.services.analysis_decisions import evaluate_analysis_decisions
        log = evaluate_analysis_decisions(df)
        result["analysis_decisions"] = log.to_dict()
        summary = log.to_dict()["summary"]
        logger.info("Analysis decisions: %d ran, %d skipped", summary["ran"], summary["skipped"])


class SemanticStep:
    name = "semantic_analysis"

    def __call__(self, df: pl.DataFrame, result: dict[str, Any]) -> None:
        from app.services.semantic_inference import analyze_semantic_structure
        analysis = analyze_semantic_structure(df)
        result["semantic_analysis"] = analysis.to_dict()
        logger.info(
            "Semantic analysis: Domain=%s (%.0f%% confidence), %d analytical cols, %d suggestions",
            analysis.domain.primary_domain,
            analysis.domain.confidence * 100,
            len(analysis.analytical_columns),
            len(analysis.suggested_pairs),
        )


class FeatureEngineeringStep:
    name = "feature_engineering"

    def __call__(self, df: pl.DataFrame, result: dict[str, Any]) -> None:
        from app.services.feature_engineering import analyze_feature_engineering
        column_roles: dict[str, str] = {}
        sem = result.get("semantic_analysis")
        if sem and sem.get("column_roles"):
            column_roles = {k: v.get("role", "") for k, v in sem["column_roles"].items()}
        fe = analyze_feature_engineering(df, column_roles)
        result["feature_engineering"] = fe.to_dict()
        logger.info(
            "Feature engineering: %d encoding, %d scaling suggestions",
            len(fe.encoding_recommendations),
            len(fe.scaling_recommendations),
        )


class SchemaStep:
    name = "smart_schema"

    def __call__(self, df: pl.DataFrame, result: dict[str, Any]) -> None:
        from app.services.smart_schema import analyze_smart_schema
        schema = analyze_smart_schema(df)
        result["smart_schema"] = schema.to_dict()
        logger.info(
            "Smart schema: %d corrections, %d relationships",
            len(schema.type_corrections),
            len(schema.relationships),
        )


class RecommendationsStep:
    name = "recommendations"

    def __call__(self, df: pl.DataFrame, result: dict[str, Any]) -> None:
        from app.services.recommendations import generate_recommendations
        domain = "unknown"
        sem = result.get("semantic_analysis")
        if sem and sem.get("domain"):
            domain = sem["domain"].get("primary", "unknown")
        rec = generate_recommendations(df, domain, result)
        result["recommendations"] = rec.to_dict()
        high = rec.get_high_priority()
        logger.info(
            "Recommendations: %d total, %d high priority",
            rec.to_dict()["total_count"],
            len(high),
        )


# ─── Default Pipeline ───────────────────────────────────────────────────────

DEFAULT_STEPS: Sequence[AnalysisStep] = [
    ConfidenceStep(),
    DecisionsStep(),
    SemanticStep(),
    FeatureEngineeringStep(),
    SchemaStep(),
    RecommendationsStep(),
]


def run_pipeline(
    df: pl.DataFrame,
    result: dict[str, Any],
    steps: Sequence[AnalysisStep] = DEFAULT_STEPS,
) -> PipelineResult:
    """
    Execute every step in order.  Each step enriches *result* in-place.

    Only ``SAFE_DATA_ERRORS`` are caught — code bugs (ImportError,
    AttributeError, etc.) propagate immediately so they surface during
    development instead of being silently swallowed.
    """
    outcome = PipelineResult()
    for step in steps:
        try:
            step(df, result)
            outcome.steps.append(StepResult(name=step.name, success=True))
        except SAFE_DATA_ERRORS as exc:
            logger.warning(
                "%s step failed: %s", step.name, exc, exc_info=True,
            )
            result[step.name] = None
            outcome.steps.append(StepResult(name=step.name, success=False, error=str(exc)))
    return outcome
