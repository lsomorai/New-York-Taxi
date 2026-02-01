"""Machine learning models for NYC Taxi Fare Prediction."""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "total_amount",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and target vector from DataFrame.

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_col: Name of target column

    Returns:
        Tuple of (X, y) arrays
    """
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for predictions.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Dictionary with R2 and RMSE metrics
    """
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    c_values: list = None,
    cv_folds: int = 3,
) -> Dict[str, Any]:
    """
    Train Support Vector Machine regressor with grid search.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        c_values: List of C values for grid search
        cv_folds: Number of cross-validation folds

    Returns:
        Dictionary with model, predictions, and metrics
    """
    if c_values is None:
        c_values = [0.5, 1.0]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="linear", cache_size=1000)),
        ]
    )

    param_grid = {"svr__C": c_values}

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_folds,
        scoring="r2",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    return {
        "model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "predictions": y_pred,
        "metrics": metrics,
    }


def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 0.1,
) -> Dict[str, Any]:
    """
    Train Ridge Regression model.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        alpha: Regularization strength

    Returns:
        Dictionary with model, predictions, and metrics
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    return {
        "model": pipeline,
        "predictions": y_pred,
        "metrics": metrics,
        "coefficients": pipeline.named_steps["ridge"].coef_,
    }


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 7,
) -> Dict[str, Any]:
    """
    Train Random Forest regressor.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        n_estimators: Number of trees
        max_depth: Maximum tree depth

    Returns:
        Dictionary with model, predictions, and metrics
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    return {
        "model": model,
        "predictions": y_pred,
        "metrics": metrics,
        "feature_importances": model.feature_importances_,
    }


def train_gbt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 50,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> Dict[str, Any]:
    """
    Train Gradient Boosting regressor.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        n_estimators: Number of boosting stages
        max_depth: Maximum tree depth
        learning_rate: Learning rate

    Returns:
        Dictionary with model, predictions, and metrics
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gbt",
                GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.8,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    return {
        "model": pipeline,
        "predictions": y_pred,
        "metrics": metrics,
        "feature_importances": pipeline.named_steps["gbt"].feature_importances_,
    }


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison table of model performances.

    Args:
        results: Dictionary mapping model names to their results

    Returns:
        DataFrame with model comparison
    """
    comparison = []
    for name, result in results.items():
        comparison.append(
            {
                "Model": name,
                "R2 Score": result["metrics"]["r2"],
                "RMSE": result["metrics"]["rmse"],
            }
        )

    return pd.DataFrame(comparison).sort_values("R2 Score", ascending=False)
