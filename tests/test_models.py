"""Tests for model functions."""

import numpy as np
import pandas as pd
import pytest

from src.models import (
    compare_models,
    evaluate_model,
    prepare_features,
    train_ridge,
    train_svm,
)


@pytest.fixture
def sample_data():
    """Create sample training and test data."""
    np.random.seed(42)

    # Create data with a linear relationship
    n_train, n_test = 100, 20
    X_train = np.random.randn(n_train, 5)
    X_test = np.random.randn(n_test, 5)

    # y = 2*x0 + 3*x1 + noise
    y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] + np.random.randn(n_train) * 0.1
    y_test = 2 * X_test[:, 0] + 3 * X_test[:, 1] + np.random.randn(n_test) * 0.1

    return X_train, y_train, X_test, y_test


@pytest.fixture
def sample_df():
    """Create sample DataFrame for feature preparation."""
    return pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.0, 6.0],
        "total_amount": [10.0, 20.0, 30.0],
    })


class TestPrepareFeatures:
    """Tests for prepare_features function."""

    def test_returns_correct_shapes(self, sample_df):
        """Test that returned arrays have correct shapes."""
        X, y = prepare_features(sample_df, ["feature1", "feature2"])

        assert X.shape == (3, 2)
        assert y.shape == (3,)

    def test_extracts_correct_values(self, sample_df):
        """Test that correct values are extracted."""
        X, y = prepare_features(sample_df, ["feature1"])

        np.testing.assert_array_equal(X.flatten(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(y, [10.0, 20.0, 30.0])


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_perfect_predictions(self):
        """Test metrics for perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        metrics = evaluate_model(y_true, y_pred)

        assert metrics["r2"] == 1.0
        assert metrics["rmse"] == 0.0

    def test_returns_correct_keys(self):
        """Test that correct metric keys are returned."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 2.9])

        metrics = evaluate_model(y_true, y_pred)

        assert "r2" in metrics
        assert "rmse" in metrics

    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])  # All off by 1

        metrics = evaluate_model(y_true, y_pred)

        assert metrics["rmse"] == 1.0


class TestTrainSvm:
    """Tests for train_svm function."""

    def test_returns_model(self, sample_data):
        """Test that a model is returned."""
        X_train, y_train, X_test, y_test = sample_data
        result = train_svm(X_train, y_train, X_test, y_test, cv_folds=2)

        assert "model" in result
        assert result["model"] is not None

    def test_returns_predictions(self, sample_data):
        """Test that predictions are returned."""
        X_train, y_train, X_test, y_test = sample_data
        result = train_svm(X_train, y_train, X_test, y_test, cv_folds=2)

        assert "predictions" in result
        assert len(result["predictions"]) == len(y_test)

    def test_returns_metrics(self, sample_data):
        """Test that metrics are returned."""
        X_train, y_train, X_test, y_test = sample_data
        result = train_svm(X_train, y_train, X_test, y_test, cv_folds=2)

        assert "metrics" in result
        assert "r2" in result["metrics"]
        assert "rmse" in result["metrics"]

    def test_reasonable_performance(self, sample_data):
        """Test that model achieves reasonable performance on linear data."""
        X_train, y_train, X_test, y_test = sample_data
        result = train_svm(X_train, y_train, X_test, y_test, cv_folds=2)

        # Should achieve high R2 on linear data
        assert result["metrics"]["r2"] > 0.8


class TestTrainRidge:
    """Tests for train_ridge function."""

    def test_returns_model(self, sample_data):
        """Test that a model is returned."""
        X_train, y_train, X_test, y_test = sample_data
        result = train_ridge(X_train, y_train, X_test, y_test)

        assert "model" in result
        assert result["model"] is not None

    def test_returns_coefficients(self, sample_data):
        """Test that coefficients are returned."""
        X_train, y_train, X_test, y_test = sample_data
        result = train_ridge(X_train, y_train, X_test, y_test)

        assert "coefficients" in result
        assert len(result["coefficients"]) == X_train.shape[1]

    def test_reasonable_performance(self, sample_data):
        """Test that model achieves reasonable performance on linear data."""
        X_train, y_train, X_test, y_test = sample_data
        result = train_ridge(X_train, y_train, X_test, y_test)

        # Should achieve high R2 on linear data
        assert result["metrics"]["r2"] > 0.8


class TestCompareModels:
    """Tests for compare_models function."""

    def test_returns_dataframe(self):
        """Test that a DataFrame is returned."""
        results = {
            "Model A": {"metrics": {"r2": 0.9, "rmse": 1.0}},
            "Model B": {"metrics": {"r2": 0.8, "rmse": 1.5}},
        }

        comparison = compare_models(results)

        assert isinstance(comparison, pd.DataFrame)

    def test_contains_all_models(self):
        """Test that all models are included."""
        results = {
            "Model A": {"metrics": {"r2": 0.9, "rmse": 1.0}},
            "Model B": {"metrics": {"r2": 0.8, "rmse": 1.5}},
        }

        comparison = compare_models(results)

        assert len(comparison) == 2
        assert "Model A" in comparison["Model"].values
        assert "Model B" in comparison["Model"].values

    def test_sorted_by_r2(self):
        """Test that results are sorted by R2 descending."""
        results = {
            "Model A": {"metrics": {"r2": 0.7, "rmse": 2.0}},
            "Model B": {"metrics": {"r2": 0.9, "rmse": 1.0}},
        }

        comparison = compare_models(results)

        assert comparison.iloc[0]["Model"] == "Model B"
        assert comparison.iloc[1]["Model"] == "Model A"
