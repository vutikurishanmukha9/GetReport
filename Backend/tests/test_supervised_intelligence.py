import time
import numpy as np
import polars as pl
import pytest

from app.services.supervised import (
    TaskType,
    SupervisedAutoMLService,
    TargetInferenceService,
    AutoMLMatrixPreprocessor,
    AutoMLResult
)

@pytest.fixture
def sample_classification_df() -> pl.DataFrame:
    np.random.seed(42)
    n = 200
    age = np.random.randint(18, 70, size=n)
    tenure = np.random.randint(1, 10, size=n)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n)
    # Target strongly driven by tenure and charges
    logits = (monthly_charges * 0.05) - (tenure * 0.6) - 1.0
    probs = 1.0 / (1.0 + np.exp(-logits))
    churn = (np.random.rand(n) < probs).astype(int)

    return pl.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "age": age,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "contract": ["month-to-month" if i % 2 == 0 else "two-year" for i in range(n)],
        "churn": churn
    })

@pytest.fixture
def sample_regression_df() -> pl.DataFrame:
    np.random.seed(42)
    n = 200
    square_feet = np.random.uniform(500, 3500, size=n)
    bedrooms = np.random.randint(1, 5, size=n)
    location_score = np.random.uniform(1.0, 10.0, size=n)
    price = (square_feet * 150.0) + (bedrooms * 20000.0) + (location_score * 10000.0) + np.random.normal(0, 5000, size=n)

    return pl.DataFrame({
        "property_id": [f"PROP_{i:04d}" for i in range(n)],
        "square_feet": square_feet,
        "bedrooms": bedrooms,
        "location_score": location_score,
        "price": price
    })

def test_target_detector_binary(sample_classification_df):
    detector = TargetInferenceService()
    task = detector.infer_task_type(sample_classification_df, "churn")
    assert task == TaskType.BINARY_CLASSIFICATION

def test_target_detector_regression(sample_regression_df):
    detector = TargetInferenceService()
    task = detector.infer_task_type(sample_regression_df, "price")
    assert task == TaskType.REGRESSION

def test_target_detector_auto_candidate_detection(sample_classification_df, sample_regression_df):
    detector = TargetInferenceService()
    candidate_c = detector.detect_candidate_target(sample_classification_df)
    assert candidate_c == "churn"

    candidate_r = detector.detect_candidate_target(sample_regression_df)
    assert candidate_r == "price"

def test_preprocessor_anti_leakage(sample_regression_df):
    # Add a leaky duplicate column with correlation 1.0
    leaky_df = sample_regression_df.with_columns(
        (pl.col("price") * 1.0001).alias("price_clone")
    )
    preprocessor = AutoMLMatrixPreprocessor(random_state=42)
    X_train, X_test, y_train, y_test, feature_names, _ = preprocessor.prepare_and_split(
        leaky_df, "price", TaskType.REGRESSION
    )
    # Leakage column and property_id must be stripped
    assert "price_clone" not in feature_names
    assert "property_id" not in feature_names
    assert "square_feet" in feature_names

def test_automl_binary_classification_end_to_end(sample_classification_df):
    service = SupervisedAutoMLService(random_state=42)
    result = service.run_automl(sample_classification_df, target_column="churn")

    assert result.ran is True
    assert result.task_type == "binary_classification"
    assert result.best_model_name in ["HistGradientBoostingClassifier", "RandomForestClassifier", "LogisticRegression"]
    assert len(result.leaderboard) == 3
    assert result.primary_metric_name == "accuracy"
    assert result.primary_metric_value is not None
    assert result.primary_metric_value > 0.60
    assert len(result.key_drivers) > 0
    # Tenure and monthly charges should be top drivers
    driver_features = [d.feature for d in result.key_drivers]
    assert "tenure" in driver_features or "monthly_charges" in driver_features
    assert "Supervised Binary Classification on 'churn'" in result.summary

def test_automl_regression_end_to_end(sample_regression_df):
    service = SupervisedAutoMLService(random_state=42)
    result = service.run_automl(sample_regression_df, target_column="price")

    assert result.ran is True
    assert result.task_type == "regression"
    assert result.primary_metric_name == "r2_score"
    assert result.primary_metric_value is not None
    assert result.primary_metric_value > 0.85
    assert len(result.leaderboard) == 3
    assert len(result.key_drivers) > 0
    assert result.key_drivers[0].feature in ["square_feet", "bedrooms", "location_score"]

def test_automl_graceful_bypass_small_data():
    service = SupervisedAutoMLService(random_state=42)
    small_df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    result = service.run_automl(small_df, target_column="y")

    assert result.ran is False
    assert "at least 15 rows" in (result.reason or "")

def test_automl_no_target_bypass():
    service = SupervisedAutoMLService(random_state=42)
    generic_df = pl.DataFrame({
        "col_a": list(range(30)),
        "col_b": [i * 2 for i in range(30)],
        "col_c": [f"val_{i % 3}" for i in range(30)]
    })
    result = service.run_automl(generic_df, target_column=None)
    assert result.ran is False
    assert "No target column specified" in (result.reason or "")

def test_automl_performance_benchmark():
    np.random.seed(42)
    n = 10000
    df = pl.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
        "feat3": np.random.randint(0, 10, size=n),
        "churn": (np.random.randn(n) > 0).astype(int)
    })
    service = SupervisedAutoMLService(random_state=42)
    t0 = time.perf_counter()
    result = service.run_automl(df, target_column="churn")
    elapsed = time.perf_counter() - t0

    assert result.ran is True
    assert elapsed < 2.0, f"AutoML on 10,000 rows took {elapsed:.2f}s, exceeding 2.0s budget"
