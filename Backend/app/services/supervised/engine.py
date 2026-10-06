import time
import logging
from typing import Optional
import polars as pl

from .schemas import TaskType, AutoMLResult
from .target_detector import TargetInferenceService
from .preprocessor import AutoMLMatrixPreprocessor
from .trainer import AutoMLModelTrainer
from .driver_analyzer import KeyDriverAnalyzer

logger = logging.getLogger(__name__)

class SupervisedAutoMLService:
    """
    Orchestrates automated target detection, data hygiene/anti-leakage preprocessing,
    fast model leaderboard benchmarking, and key driver attribution.
    """

    def __init__(
        self,
        target_detector: Optional[TargetInferenceService] = None,
        preprocessor: Optional[AutoMLMatrixPreprocessor] = None,
        trainer: Optional[AutoMLModelTrainer] = None,
        driver_analyzer: Optional[KeyDriverAnalyzer] = None,
        random_state: int = 42
    ):
        self.target_detector = target_detector or TargetInferenceService()
        self.preprocessor = preprocessor or AutoMLMatrixPreprocessor(random_state=random_state)
        self.trainer = trainer or AutoMLModelTrainer(random_state=random_state)
        self.driver_analyzer = driver_analyzer or KeyDriverAnalyzer()

    def run_automl(
        self,
        df: pl.DataFrame,
        target_column: Optional[str] = None
    ) -> AutoMLResult:
        start_time = time.perf_counter()

        # Guard: minimal dimensions
        if df is None or len(df) < 15 or len(df.columns) < 2:
            return AutoMLResult(
                ran=False,
                reason="Dataset requires at least 15 rows and 2 columns for supervised learning.",
                total_execution_time_sec=time.perf_counter() - start_time
            )

        # 1. Resolve target column
        target_col = target_column
        if not target_col or target_col not in df.columns:
            target_col = self.target_detector.detect_candidate_target(df)

        if not target_col:
            return AutoMLResult(
                ran=False,
                reason="No target column specified and no obvious business outcome column detected.",
                total_execution_time_sec=time.perf_counter() - start_time
            )

        # 2. Infer task type
        task_type = self.target_detector.infer_task_type(df, target_col)
        if task_type == TaskType.UNKNOWN:
            return AutoMLResult(
                ran=False,
                target_column=target_col,
                reason=f"Target column '{target_col}' has ambiguous cardinality or unsupported distribution.",
                total_execution_time_sec=time.perf_counter() - start_time
            )

        # 3. Preprocess and split
        try:
            X_train, X_test, y_train, y_test, feature_names, _ = self.preprocessor.prepare_and_split(
                df, target_col, task_type
            )
        except Exception as e:
            logger.info(f"Supervised preprocessing bypassed for '{target_col}': {e}")
            return AutoMLResult(
                ran=False,
                target_column=target_col,
                task_type=task_type.value,
                reason=f"Preprocessing check: {str(e)}",
                total_execution_time_sec=time.perf_counter() - start_time
            )

        # 4. Train and benchmark leaderboard
        try:
            best_model, leaderboard, best_metrics = self.trainer.train_and_evaluate(
                X_train, X_test, y_train, y_test, task_type
            )
        except Exception as e:
            logger.warning(f"Supervised model training failed for '{target_col}': {e}")
            return AutoMLResult(
                ran=False,
                target_column=target_col,
                task_type=task_type.value,
                reason=f"Model training error: {str(e)}",
                total_execution_time_sec=time.perf_counter() - start_time
            )

        # 5. Extract key drivers
        try:
            key_drivers = self.driver_analyzer.analyze_drivers(
                best_model, X_train, y_train, feature_names, task_type
            )
        except Exception as e:
            logger.warning(f"Key driver extraction fallback: {e}")
            key_drivers = []

        # 6. Synthesize narrative summary
        best_entry = leaderboard[0]
        summary = self.driver_analyzer.synthesize_summary(
            target_col=target_col,
            task_type=task_type,
            best_model_name=best_entry.model_name,
            primary_metric_name=best_entry.primary_metric_name,
            primary_metric_value=best_entry.primary_metric_value,
            key_drivers=key_drivers
        )

        total_time = time.perf_counter() - start_time

        return AutoMLResult(
            ran=True,
            target_column=target_col,
            task_type=task_type.value,
            best_model_name=best_entry.model_name,
            primary_metric_name=best_entry.primary_metric_name,
            primary_metric_value=best_entry.primary_metric_value,
            metrics=best_metrics,
            leaderboard=leaderboard,
            key_drivers=key_drivers,
            summary=summary,
            total_execution_time_sec=total_time
        )
