import time
from typing import Tuple, List, Dict, Any
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error
)

from .schemas import TaskType, LeaderboardEntry

class AutoMLModelTrainer:
    """
    Trains and evaluates fast baseline models (HistGradientBoosting, RandomForest,
    Linear baselines) to construct a performance leaderboard.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        task_type: TaskType
    ) -> Tuple[Any, List[LeaderboardEntry], Dict[str, float]]:
        """
        Trains model candidates, evaluates on holdout test set, and returns:
        (best_fitted_model, leaderboard, best_metrics)
        """
        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            return self._train_classification(X_train, X_test, y_train, y_test, task_type)
        elif task_type == TaskType.REGRESSION:
            return self._train_regression(X_train, X_test, y_train, y_test)
        else:
            raise ValueError(f"Unsupported task type for training: {task_type}")

    def _train_classification(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        task_type: TaskType
    ) -> Tuple[Any, List[LeaderboardEntry], Dict[str, float]]:
        candidates = [
            ("HistGradientBoostingClassifier", HistGradientBoostingClassifier(random_state=self.random_state)),
            ("RandomForestClassifier", RandomForestClassifier(n_estimators=15, max_depth=6, random_state=self.random_state, n_jobs=1)),
            ("LogisticRegression", LogisticRegression(max_iter=400, random_state=self.random_state)),
        ]

        leaderboard: List[LeaderboardEntry] = []
        fitted_models: List[Tuple[str, Any, Dict[str, float]]] = []

        is_binary = (task_type == TaskType.BINARY_CLASSIFICATION) and (len(np.unique(y_test)) == 2)

        for name, model in candidates:
            try:
                t0 = time.perf_counter()
                model.fit(X_train, y_train)
                train_time = time.perf_counter() - t0

                y_pred = model.predict(X_test)
                acc = float(accuracy_score(y_test, y_pred))
                f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

                metrics: Dict[str, float] = {
                    "accuracy": acc,
                    "f1_score": f1
                }

                if is_binary and hasattr(model, "predict_proba"):
                    try:
                        y_prob = model.predict_proba(X_test)[:, 1]
                        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
                    except Exception:
                        pass

                entry = LeaderboardEntry(
                    model_name=name,
                    primary_metric_name="accuracy",
                    primary_metric_value=acc,
                    all_metrics=metrics,
                    train_time_sec=train_time
                )
                leaderboard.append(entry)
                fitted_models.append((name, model, metrics))
            except Exception:
                continue

        if not leaderboard:
            raise RuntimeError("All classification models failed to fit.")

        # Sort leaderboard descending by primary metric
        leaderboard.sort(key=lambda x: x.primary_metric_value, reverse=True)
        best_name = leaderboard[0].model_name

        best_tuple = next(t for t in fitted_models if t[0] == best_name)
        best_model = best_tuple[1]
        best_metrics = best_tuple[2]

        return best_model, leaderboard, best_metrics

    def _train_regression(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Any, List[LeaderboardEntry], Dict[str, float]]:
        candidates = [
            ("HistGradientBoostingRegressor", HistGradientBoostingRegressor(random_state=self.random_state)),
            ("RandomForestRegressor", RandomForestRegressor(n_estimators=15, max_depth=6, random_state=self.random_state, n_jobs=1)),
            ("Ridge", Ridge(alpha=1.0, random_state=self.random_state)),
        ]

        leaderboard: List[LeaderboardEntry] = []
        fitted_models: List[Tuple[str, Any, Dict[str, float]]] = []

        for name, model in candidates:
            try:
                t0 = time.perf_counter()
                model.fit(X_train, y_train)
                train_time = time.perf_counter() - t0

                y_pred = model.predict(X_test)
                r2 = float(r2_score(y_test, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae = float(mean_absolute_error(y_test, y_pred))

                metrics: Dict[str, float] = {
                    "r2_score": r2,
                    "rmse": rmse,
                    "mae": mae
                }

                entry = LeaderboardEntry(
                    model_name=name,
                    primary_metric_name="r2_score",
                    primary_metric_value=r2,
                    all_metrics=metrics,
                    train_time_sec=train_time
                )
                leaderboard.append(entry)
                fitted_models.append((name, model, metrics))
            except Exception:
                continue

        if not leaderboard:
            raise RuntimeError("All regression models failed to fit.")

        # Sort descending by R2
        leaderboard.sort(key=lambda x: x.primary_metric_value, reverse=True)
        best_name = leaderboard[0].model_name

        best_tuple = next(t for t in fitted_models if t[0] == best_name)
        best_model = best_tuple[1]
        best_metrics = best_tuple[2]

        return best_model, leaderboard, best_metrics
