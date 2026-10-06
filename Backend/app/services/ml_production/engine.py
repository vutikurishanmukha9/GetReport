from typing import Optional, Dict, Any, Tuple
import polars as pl

from .schemas import ExecutiveMLSection, ModelArtifactMetadata, BatchPredictionResult
from .report_synthesizer import ExecutiveMLReportSynthesizer
from .serializer import ModelArtifactSerializer
from .batch_scorer import BatchInferenceEngine

class MLProductionService:
    """
    Coordinates Executive Report Synthesis, Model Artifact Packaging,
    and High-Throughput Batch Scoring.
    """

    def __init__(
        self,
        report_synthesizer: Optional[ExecutiveMLReportSynthesizer] = None,
        serializer: Optional[ModelArtifactSerializer] = None,
        batch_scorer: Optional[BatchInferenceEngine] = None,
    ):
        self.report_synthesizer = report_synthesizer or ExecutiveMLReportSynthesizer()
        self.serializer = serializer or ModelArtifactSerializer()
        self.batch_scorer = batch_scorer or BatchInferenceEngine()

    def build_report_section(
        self,
        unsupervised_data: Optional[Dict[str, Any]] = None,
        supervised_data: Optional[Dict[str, Any]] = None,
    ) -> ExecutiveMLSection:
        """
        Translates machine learning outputs into executive takeaways and summary tables.
        """
        return self.report_synthesizer.synthesize_section(
            unsupervised_data=unsupervised_data,
            supervised_data=supervised_data,
        )

    def save_model(
        self,
        model: Any,
        metadata: ModelArtifactMetadata,
        output_path: str,
        preprocessor_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Serializes a trained model bundle for production archiving and inference.
        """
        return self.serializer.save_artifact(
            model=model,
            metadata=metadata,
            output_path=output_path,
            preprocessor_meta=preprocessor_meta,
        )

    def score_batch(
        self,
        artifact_path: str,
        df: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, BatchPredictionResult]:
        """
        Loads a serialized model artifact and scores a new incoming dataset.
        """
        model, metadata, preprocessor_meta = self.serializer.load_artifact(artifact_path)
        return self.batch_scorer.score_dataframe(
            model=model,
            metadata=metadata,
            df=df,
            preprocessor_meta=preprocessor_meta,
        )
