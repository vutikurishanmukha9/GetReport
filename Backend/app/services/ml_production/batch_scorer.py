from typing import Tuple, Dict, Any, List
import numpy as np
import polars as pl

from .schemas import ModelArtifactMetadata, BatchPredictionResult

class BatchInferenceEngine:
    """
    Performs high-throughput schema validation and vectorized batch prediction
    on unseen tabular datasets using saved production model artifacts.
    """

    def score_dataframe(
        self,
        model: Any,
        metadata: ModelArtifactMetadata,
        df: pl.DataFrame,
        preprocessor_meta: Dict[str, Any]
    ) -> Tuple[pl.DataFrame, BatchPredictionResult]:
        if df is None or len(df) == 0:
            return df, BatchPredictionResult(
                success=False,
                total_rows_scored=0,
                predictions_col="",
                error_message="Input dataset is empty."
            )

        # 1. Validate required feature columns
        missing_cols = [col for col in metadata.feature_names if col not in df.columns]
        if missing_cols:
            return df, BatchPredictionResult(
                success=False,
                total_rows_scored=0,
                predictions_col="",
                error_message=f"Missing {len(missing_cols)} required feature columns: {missing_cols[:5]}"
            )

        # 2. Extract feature matrix
        feature_arrays: List[np.ndarray] = []
        for col in metadata.feature_names:
            series = df[col]
            dtype = series.dtype

            if dtype in (
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64
            ):
                arr = series.to_numpy().astype(np.float64)
                arr = np.where(np.isnan(arr), 0.0, arr)
                feature_arrays.append(arr.reshape(-1, 1))
            else:
                # String / Categorical fallback encoding
                val_counts = series.value_counts()
                top_vals = [row[0] for row in val_counts.rows() if row[0] is not None][:30]
                val_map = {val: idx + 1 for idx, val in enumerate(top_vals)}
                encoded = np.array([val_map.get(v, 0) for v in series.to_list()], dtype=np.float64)
                feature_arrays.append(encoded.reshape(-1, 1))

        X = np.hstack(feature_arrays)

        # 3. Predict
        try:
            preds = model.predict(X)
        except Exception as e:
            return df, BatchPredictionResult(
                success=False,
                total_rows_scored=0,
                predictions_col="",
                error_message=f"Model prediction failed: {str(e)}"
            )

        pred_col_name = f"predicted_{metadata.target_column or 'target'}"
        enriched_df = df.with_columns(pl.Series(name=pred_col_name, values=preds))
        prob_col_name = None

        # Predict probabilities if binary classification
        if metadata.task_type == "binary_classification" and hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X)
                if probs.shape[1] >= 2:
                    prob_col_name = f"{pred_col_name}_probability"
                    enriched_df = enriched_df.with_columns(
                        pl.Series(name=prob_col_name, values=probs[:, 1])
                    )
            except Exception:
                pass

        return enriched_df, BatchPredictionResult(
            success=True,
            total_rows_scored=len(df),
            predictions_col=pred_col_name,
            probabilities_col=prob_col_name
        )
