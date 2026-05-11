from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from qfactor_penny.export_qnn_hardware_subset import export_qnn_hardware_subset
from qfactor_penny.ibm_hardware_inference import run_hardware_inference
from qfactor_penny.ibm_transpile_audit import run_transpile_audit
from qfactor_penny.make_hardware_report import make_hardware_report
from qfactor_penny.prepare_data import prepare_dataset
from qfactor_penny.qiskit_qnn import qiskit_statevector_score
from qfactor_penny.qnn import fit_predict_qnn

LOCAL_PATH_MARKER = "/" + "Users/"


def test_qnn_result_exposes_frozen_parameters():
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
        shot_sensitivity_samples=0,
        shot_sensitivity_shots=0,
    )
    assert result.weights.shape == (1, 4, 3)
    assert np.isfinite(result.weights).all()
    assert np.isfinite(result.bias)


def test_frozen_qnn_export_contains_required_fields_and_no_token(tmp_path, monkeypatch):
    monkeypatch.setenv("QISKIT_IBM_TOKEN", "secret-token-that-must-not-appear")
    dataset = tmp_path / "qfactor_dataset.csv"
    prepare_dataset(tmp_path / "missing.csv", dataset)
    config_path = tmp_path / "config.json"
    output = tmp_path / "results_hardware" / "frozen_qnn_subset.json"
    config = {
        "dataset_path": str(dataset),
        "results_dir": str(tmp_path / "results"),
        "min_train_dates": 8,
        "validation_dates": 3,
        "purge_trading_days": 5,
        "max_splits": 1,
        "feature_count": 4,
        "qnn_epochs": 1,
        "qnn_learning_rate": 0.01,
        "qnn_patience": 1,
        "mlp_hidden_units": 4,
        "transaction_cost": 0.0005,
        "random_state": 42,
        "seeds": [42],
        "feature_selection": {"mode": "cross_sectional_aware", "min_cross_sectional_std_quantile": 0.25},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifact = export_qnn_hardware_subset(
        config,
        output=output,
        config_path=config_path,
        max_models=1,
        max_dates_per_model=1,
    )
    text = output.read_text(encoding="utf-8")
    model = artifact["qnn_models"][0]
    sample = model["samples"][0]
    assert artifact["config_hash"]
    assert artifact["dataset_hash"]
    assert model["weights"]
    assert isinstance(model["bias"], float)
    assert model["selected_features"]
    assert {"split_id", "seed", "samples"}.issubset(model)
    assert {"date", "ticker", "feature_values", "analytic_score"}.issubset(sample)
    assert "secret-token-that-must-not-appear" not in text
    assert "QISKIT_IBM_TOKEN" not in text


def test_qiskit_statevector_score_matches_pennylane_qnn_when_qiskit_is_installed():
    pytest.importorskip("qiskit")
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
        x_train[:1],
        selected_features=["a", "b", "c", "d"],
        epochs=1,
        learning_rate=0.01,
        patience=1,
        random_state=42,
        shot_sensitivity_samples=0,
        shot_sensitivity_shots=0,
    )
    qiskit_score = qiskit_statevector_score(x_train[0], result.weights, result.bias)
    assert qiskit_score == pytest.approx(float(result.score[0]), abs=1e-8)


def test_hardware_audit_clis_skip_cleanly_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    monkeypatch.delenv("QISKIT_IBM_INSTANCE", raising=False)
    monkeypatch.chdir(tmp_path)
    frozen = tmp_path / "frozen_qnn_subset.json"
    frozen.write_text(json.dumps(_minimal_frozen_artifact()), encoding="utf-8")
    transpile_csv = tmp_path / "ibm_transpilation_audit.csv"
    hardware_csv = tmp_path / "ibm_hardware_scores.csv"
    transpile_rows = run_transpile_audit(input_path=frozen, output=transpile_csv)
    hardware_rows = run_hardware_inference(input_path=frozen, output=hardware_csv, resilience_levels=[0])
    assert transpile_csv.exists()
    assert hardware_csv.exists()
    assert {row["status"] for row in transpile_rows} == {"skipped"}
    assert {row["status"] for row in hardware_rows} == {"skipped"}
    assert pd.read_csv(transpile_csv)["error_type"].notna().all()
    assert pd.read_csv(hardware_csv)["error_type"].notna().all()


def test_hardware_report_contains_guardrails_and_no_local_paths(tmp_path):
    pd.DataFrame(
        [
            {
                "status": "skipped",
                "error_type": "MissingCredentials",
                "error_message": "QISKIT_IBM_TOKEN is not set",
                "model_id": "split_00__seed_42",
                "split_id": "split_00",
                "seed": 42,
                "date": "2024-01-02",
                "ticker": "XLK",
                "sample_index": 0,
            }
        ]
    ).to_csv(tmp_path / "ibm_hardware_scores.csv", index=False)
    output = make_hardware_report(tmp_path)
    text = output.read_text(encoding="utf-8")
    assert "systems diagnostic" in text
    assert "does not claim quantum advantage" in text
    assert LOCAL_PATH_MARKER not in text


def test_hardware_report_handles_failed_and_successful_rows_together(tmp_path):
    pd.DataFrame(
        [
            {
                "status": "success",
                "error_type": "",
                "error_message": "",
                "backend_name": "ibm_rensselaer",
                "job_id": "job-ok",
                "optimization_level": 0,
                "resilience_level": 0,
                "requested_precision": 0.1,
                "shots": 100,
                "model_id": "split_00__seed_42",
                "split_id": "split_00",
                "seed": 42,
                "date": "2024-01-02",
                "ticker": ticker,
                "sample_index": index,
                "analytic_score": analytic,
                "hardware_score": hardware,
            }
            for index, (ticker, analytic, hardware) in enumerate(
                [
                    ("XLB", 0.3, 0.1),
                    ("XLK", 0.2, 0.4),
                    ("XLP", -0.1, -0.3),
                ]
            )
        ]
        + [
            {
                "status": "failed",
                "error_type": "RuntimeJobFailureError",
                "error_message": "HAL error 9604: Failed to set configuration on HAL components",
                "backend_name": "ibm_rensselaer",
                "job_id": "job-failed",
                "optimization_level": 1,
                "resilience_level": 0,
                "requested_precision": 0.1,
                "shots": 100,
                "model_id": "split_00__seed_42",
                "split_id": "split_00",
                "seed": 42,
                "date": "2024-01-02",
                "ticker": "XLV",
                "sample_index": 3,
                "analytic_score": np.nan,
                "hardware_score": np.nan,
            }
        ]
    ).to_csv(tmp_path / "ibm_hardware_scores.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "success",
                "backend_name": "ibm_rensselaer",
                "optimization_level": 0,
                "ticker": "XLB",
                "pre_depth": 8,
                "post_depth": 26,
                "pre_two_qubit_gate_count": 4,
                "post_two_qubit_gate_count": 9,
                "post_swap_count": 0,
            }
        ]
    ).to_csv(tmp_path / "ibm_transpilation_audit.csv", index=False)
    output = make_hardware_report(tmp_path)
    text = output.read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "hardware_run_manifest.json").read_text(encoding="utf-8"))
    assert "Non-success hardware rows were also recorded" in text
    assert "9604" in text
    assert manifest["artifact_hashes"]["ibm_hardware_scores.csv"]
    assert "secret" not in json.dumps(manifest)
    assert LOCAL_PATH_MARKER not in text
    assert LOCAL_PATH_MARKER not in json.dumps(manifest)


def _minimal_frozen_artifact() -> dict:
    return {
        "schema_version": 1,
        "qnn_models": [
            {
                "model_id": "split_00__seed_42",
                "split_id": "split_00",
                "seed": 42,
                "selected_features": ["a", "b", "c", "d"],
                "weights": np.zeros((1, 4, 3)).tolist(),
                "bias": 0.0,
                "n_qubits": 4,
                "n_layers": 1,
                "trainable_parameter_count": 13,
                "samples": [
                    {
                        "sample_index": 0,
                        "date": "2024-01-02",
                        "ticker": "XLK",
                        "label": 1.0,
                        "feature_values": [0.1, 0.2, 0.3, 0.4],
                        "analytic_score": 0.9,
                        "forward_return_5d": 0.01,
                        "spy_forward_return_5d": 0.002,
                        "excess_return_5d": 0.008,
                        "is_synthetic": True,
                        "data_source": "unit",
                    }
                ],
            }
        ],
    }
