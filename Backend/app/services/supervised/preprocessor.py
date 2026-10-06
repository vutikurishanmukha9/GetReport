from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer

from .schemas import TaskType

ID_PATTERNS = ["id", "uuid", "guid", "key", "index", "hash", "row_num"]

class AutoMLMatrixPreprocessor:
    """
    Handles feature-target matrix preparation, anti-leakage filtering,
    categorical encoding, missing value imputation, and train/test splitting.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def prepare_and_split(
        self,
        df: pl.DataFrame,
        target_col: str,
        task_type: TaskType,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """
        Prepares X_train, X_test, y_train, y_test, feature_names, and metadata.
        Raises ValueError if data is insufficient after cleaning.
        """
        # 1. Filter out rows where target is null
        valid_df = df.filter(pl.col(target_col).is_not_null())
        if len(valid_df) < 15:
            raise ValueError(f"Insufficient non-null rows for target '{target_col}' (found {len(valid_df)}).")

        # Subsample large datasets for memory efficiency and sub-second performance
        if len(valid_df) > 1500:
            valid_df = valid_df.sample(n=1500, seed=self.random_state)

        # 2. Extract and encode target
        y_raw = valid_df[target_col].to_numpy()
        meta: Dict[str, Any] = {"target_classes": None}

        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            meta["target_classes"] = [str(c) for c in le.classes_]
            # Ensure at least 2 distinct classes exist in target
            if len(np.unique(y)) < 2:
                raise ValueError("Target contains fewer than 2 distinct classes.")
        elif task_type == TaskType.REGRESSION:
            y = y_raw.astype(np.float64)
            # Filter non-finite targets if any
            finite_mask = np.isfinite(y)
            if np.sum(finite_mask) < 15:
                raise ValueError("Insufficient finite values for regression target.")
            if not np.all(finite_mask):
                valid_df = valid_df.filter(pl.Series(finite_mask))
                y = y[finite_mask]
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        # 3. Select and filter candidate features
        candidate_cols: List[str] = []
        for col in valid_df.columns:
            if col == target_col:
                continue
            col_lower = col.lower().strip()
            # Drop ID tokens
            if any(id_token == col_lower or f"_{id_token}" in col_lower or f"{id_token}_" in col_lower for id_token in ID_PATTERNS):
                continue
            # Drop date/time types
            dtype = valid_df[col].dtype
            if dtype in (pl.Date, pl.Datetime, pl.Time, pl.Duration):
                continue
            candidate_cols.append(col)

        if not candidate_cols:
            raise ValueError("No valid predictive features remaining after filtering ID/timestamp columns.")

        # 4. Process features (numerics & categoricals)
        feature_arrays: List[np.ndarray] = []
        final_feature_names: List[str] = []

        for col in candidate_cols:
            series = valid_df[col]
            dtype = series.dtype

            # Numeric handling
            if dtype in (
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64
            ):
                arr = series.to_numpy().astype(np.float64)
                # Check for constant column
                valid_vals = arr[np.isfinite(arr)]
                if len(valid_vals) == 0 or np.std(valid_vals) < 1e-7:
                    continue

                # Leakage check: perfect correlation with continuous target
                if task_type == TaskType.REGRESSION and len(valid_vals) == len(arr):
                    std_arr = np.std(arr)
                    std_y = np.std(y)
                    if std_arr > 1e-7 and std_y > 1e-7:
                        corr = np.corrcoef(arr, y)[0, 1]
                        if np.abs(corr) > 0.99:
                            continue  # Potential leakage or duplicate

                # Impute missing values with median
                median_val = float(np.nanmedian(arr)) if len(valid_vals) > 0 else 0.0
                arr = np.where(np.isnan(arr), median_val, arr)

                feature_arrays.append(arr.reshape(-1, 1))
                final_feature_names.append(col)

            # Categorical handling
            elif dtype in (pl.Utf8, pl.Categorical, pl.Boolean):
                n_unique = series.drop_nulls().n_unique()
                # Ignore very high cardinality strings (like descriptions, URLs, names)
                if n_unique < 2 or n_unique > 30:
                    continue

                # Frequency / ordinal encode
                val_counts = series.value_counts()
                top_vals = [row[0] for row in val_counts.rows() if row[0] is not None][:30]
                val_map = {val: idx + 1 for idx, val in enumerate(top_vals)}

                encoded = np.array([val_map.get(v, 0) for v in series.to_list()], dtype=np.float64)
                feature_arrays.append(encoded.reshape(-1, 1))
                final_feature_names.append(col)

        if not feature_arrays:
            raise ValueError("No predictive features with sufficient variance remaining.")

        X = np.hstack(feature_arrays).astype(np.float32)

        # 5. Robust scaling for consistent model stability
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)

        # 6. Stratified / Standard train-test split
        stratify = None
        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            _, counts = np.unique(y, return_counts=True)
            if np.min(counts) >= 2:
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify
        )

        return X_train, X_test, y_train, y_test, final_feature_names, meta
