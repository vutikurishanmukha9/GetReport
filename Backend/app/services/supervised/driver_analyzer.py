from typing import List, Any
import numpy as np
from sklearn.inspection import permutation_importance
from .schemas import KeyDriver, TaskType

class KeyDriverAnalyzer:
    """
    Extracts feature importance, calculates impact directionality,
    and synthesizes human-readable driver attribution takeaways.
    """

    def analyze_drivers(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        task_type: TaskType,
        max_drivers: int = 5
    ) -> List[KeyDriver]:
        if len(feature_names) == 0:
            return []

        # 1. Compute raw importances
        raw_importances = None
        if hasattr(model, "feature_importances_"):
            raw_importances = np.array(model.feature_importances_, dtype=np.float64)
        elif hasattr(model, "coef_"):
            coef = np.array(model.coef_, dtype=np.float64)
            if coef.ndim > 1:
                raw_importances = np.mean(np.abs(coef), axis=0)
            else:
                raw_importances = np.abs(coef)

        if raw_importances is None or np.sum(raw_importances) < 1e-9:
            # Fallback to permutation importance on a small sample of X_train
            try:
                sample_idx = np.random.choice(len(X_train), size=min(100, len(X_train)), replace=False)
                res = permutation_importance(model, X_train[sample_idx], y_train[sample_idx], n_repeats=3, random_state=42)
                raw_importances = np.maximum(0, res.importances_mean)
            except Exception:
                raw_importances = np.ones(len(feature_names), dtype=np.float64)

        # Normalize to 100%
        total_imp = np.sum(raw_importances)
        if total_imp > 1e-9:
            norm_importances = (raw_importances / total_imp) * 100.0
        else:
            norm_importances = np.full(len(feature_names), 100.0 / len(feature_names))

        # 2. Sort features descending by importance
        sorted_indices = np.argsort(norm_importances)[::-1]

        drivers: List[KeyDriver] = []
        for idx in sorted_indices[:max_drivers]:
            feat = feature_names[idx]
            imp_pct = float(norm_importances[idx])

            # Calculate directionality
            feat_vals = X_train[:, idx]
            direction = "neutral"
            if np.std(feat_vals) > 1e-7 and np.std(y_train) > 1e-7:
                corr = np.corrcoef(feat_vals, y_train)[0, 1]
                if corr > 0.08:
                    direction = "positive"
                elif corr < -0.08:
                    direction = "negative"

            # Formulate impact description
            clean_feat = feat.replace("_", " ").title()
            if direction == "positive":
                impact_desc = f"Higher {clean_feat} increases target likelihood/values."
            elif direction == "negative":
                impact_desc = f"Higher {clean_feat} decreases target likelihood/values."
            else:
                impact_desc = f"{clean_feat} has complex non-linear predictive influence."

            drivers.append(KeyDriver(
                feature=feat,
                importance_pct=imp_pct,
                direction=direction,
                impact_description=impact_desc
            ))

        return drivers

    def synthesize_summary(
        self,
        target_col: str,
        task_type: TaskType,
        best_model_name: str,
        primary_metric_name: str,
        primary_metric_value: float,
        key_drivers: List[KeyDriver]
    ) -> str:
        task_label = task_type.value.replace("_", " ").title()
        metric_label = primary_metric_name.replace("_", " ").upper()

        if not key_drivers:
            return f"Trained baseline models on '{target_col}' ({task_label}). Best model: {best_model_name} ({metric_label}: {primary_metric_value:.3f})."

        top_driver = key_drivers[0]
        summary = (
            f"Supervised {task_label} on '{target_col}' achieved best performance via {best_model_name} "
            f"({metric_label}: {primary_metric_value:.3f}). Key driver: '{top_driver.feature}' accounts for "
            f"{top_driver.importance_pct:.1f}% of predictive power ({top_driver.impact_description})."
        )
        return summary
