"""Classical baseline models."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

CLASSICAL_MODEL_NAMES = [
    "logistic_regression",
    "ridge_linear",
    "random_forest",
    "rbf_svm",
    "small_mlp",
    "xgboost",
]


@dataclass
class TrainedScorer:
    name: str
    estimator: Any
    train_seconds: float
    param_count: int | float

    def score(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "decision_function"):
            return np.asarray(self.estimator.decision_function(x), dtype=float)
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(x)
            return np.asarray(proba[:, -1], dtype=float)
        return np.asarray(self.estimator.predict(x), dtype=float)


def mlp_parameter_count(input_dim: int, hidden_units: int) -> int:
    return int(input_dim * hidden_units + hidden_units + hidden_units * 1 + 1)


def fit_classical_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int,
    mlp_hidden_units: int,
    return_status: bool = False,
) -> list[TrainedScorer] | tuple[list[TrainedScorer], list[dict[str, object]]]:
    status_rows: list[dict[str, object]] = []
    if len(np.unique(y_train)) < 2:
        warnings.warn("Classical baselines need at least two classes; skipping all.", RuntimeWarning)
        for name in CLASSICAL_MODEL_NAMES:
            status_rows.append(
                {
                    "model": name,
                    "status": "skipped",
                    "fit_success": False,
                    "error_type": "OneClassTrainingLabels",
                    "error_message": "Classical baselines need at least two classes.",
                    "train_seconds": 0.0,
                }
            )
        if return_status:
            return [], status_rows
        return []
    specs: list[tuple[str, Any, int | float]] = [
        (
            "logistic_regression",
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state),
            float(x_train.shape[1] + 1),
        ),
        ("ridge_linear", RidgeClassifier(class_weight="balanced"), float(x_train.shape[1] + 1)),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=1,
            ),
            np.nan,
        ),
        ("rbf_svm", SVC(C=1.0, gamma="scale", class_weight="balanced"), np.nan),
        (
            "small_mlp",
            MLPClassifier(
                hidden_layer_sizes=(mlp_hidden_units,),
                activation="logistic",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.01,
                max_iter=250,
                early_stopping=len(y_train) >= 30,
                validation_fraction=0.2,
                n_iter_no_change=8,
                random_state=random_state,
            ),
            mlp_parameter_count(x_train.shape[1], mlp_hidden_units),
        ),
    ]
    try:
        from xgboost import XGBClassifier

        positives = max(1, int(np.sum(y_train == 1)))
        negatives = max(1, int(np.sum(y_train == 0)))
        specs.append(
            (
                "xgboost",
                XGBClassifier(
                    n_estimators=75,
                    max_depth=2,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    scale_pos_weight=negatives / positives,
                    random_state=random_state,
                    n_jobs=1,
                ),
                np.nan,
            )
        )
    except Exception as exc:
        warnings.warn(f"XGBoost unavailable; skipping optional baseline ({exc}).", RuntimeWarning)
        status_rows.append(
            {
                "model": "xgboost",
                "status": "skipped",
                "fit_success": False,
                "error_type": type(exc).__name__,
                "error_message": f"XGBoost unavailable; skipping optional baseline ({exc}).",
                "train_seconds": 0.0,
            }
        )

    trained: list[TrainedScorer] = []
    for name, estimator, param_count in specs:
        start = time.perf_counter()
        try:
            estimator.fit(x_train, y_train)
        except Exception as exc:
            warnings.warn(f"{name} failed to fit: {exc}; skipping.", RuntimeWarning)
            status_rows.append(
                {
                    "model": name,
                    "status": "failed",
                    "fit_success": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "train_seconds": time.perf_counter() - start,
                }
            )
            continue
        train_seconds = time.perf_counter() - start
        status_rows.append(
            {
                "model": name,
                "status": "success",
                "fit_success": True,
                "error_type": "",
                "error_message": "",
                "train_seconds": train_seconds,
            }
        )
        trained.append(
            TrainedScorer(
                name=name,
                estimator=estimator,
                train_seconds=train_seconds,
                param_count=param_count,
            )
        )
    if return_status:
        return trained, status_rows
    return trained
