from __future__ import annotations

import builtins

import numpy as np
import pytest

from qfactor_penny.models import fit_classical_models, mlp_parameter_count
from qfactor_penny.qnn import fit_predict_qnn


def test_missing_xgboost_skips_cleanly(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    x_train = np.array([[0.0, 0.1, 0.2, 0.3], [1.0, 0.9, 0.8, 0.7], [0.2, 0.2, 0.2, 0.2], [0.8, 0.7, 0.6, 0.5]])
    y_train = np.array([0, 1, 0, 1])
    with pytest.warns(RuntimeWarning, match="XGBoost unavailable"):
        models = fit_classical_models(x_train, y_train, random_state=42, mlp_hidden_units=4)
    assert {model.name for model in models}.issuperset({"logistic_regression", "ridge_linear", "small_mlp"})


def test_xgboost_is_included_when_importable():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(30, 4))
    y_train = np.array([0, 1] * 15)
    models = fit_classical_models(x_train, y_train, random_state=42, mlp_hidden_units=4)
    assert "xgboost" in {model.name for model in models}


def test_mlp_parameter_count_is_reported():
    assert mlp_parameter_count(4, 4) == 25
    assert mlp_parameter_count(4, 8) == 49


def test_pennylane_qnn_returns_finite_scores_and_parameter_count():
    pytest.importorskip("pennylane")
    x_train = np.array(
        [
            [-1.0, -0.5, 0.2, 0.1],
            [-0.8, -0.2, 0.1, 0.3],
            [0.7, 0.4, -0.1, -0.2],
            [0.9, 0.5, -0.2, -0.3],
        ],
        dtype=float,
    )
    y_train = np.array([0, 0, 1, 1])
    result = fit_predict_qnn(
        x_train,
        y_train,
        x_train,
        y_train,
        x_train,
        selected_features=["a", "b", "c", "d"],
        epochs=1,
        learning_rate=0.01,
        patience=1,
        random_state=42,
        shot_sensitivity_samples=4,
        shot_sensitivity_shots=1024,
    )
    assert result.parameter_count == 13
    assert np.isfinite(result.score).all()
    assert result.shot_sensitivity_samples == 4
    assert result.shot_sensitivity_shots == 1024
    assert np.isfinite(result.shot_mean_abs_score_diff)
