from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd

from qfactor_penny.make_report import _variant_comparison
from qfactor_penny.run_benchmark import _prediction_rows


def test_prediction_model_rank_position_is_score_rank_with_realized_rank_separate():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 4,
            "ticker": ["XLB", "XLC", "XLE", "XLF"],
            "label": [1.0, None, 0.0, 1.0],
            "realized_rank_position": [4, 3, 2, 1],
            "forward_return_5d": [0.01, 0.02, 0.03, 0.04],
            "spy_forward_return_5d": [0.005] * 4,
            "excess_return_5d": [0.005, 0.015, 0.025, 0.035],
            "is_synthetic": [False] * 4,
            "data_source": ["unit"] * 4,
        }
    )
    predictions = _prediction_rows(
        split_id="split_00",
        model="demo",
        test_frame=frame,
        scores=[0.2, 0.1, 0.2, 0.3],
        selected_features=["ret_20d"],
        train_seconds=0.0,
        inference_seconds=0.0,
        param_count=0,
        seed=42,
    )
    assert "rank_position" not in predictions.columns
    assert predictions.sort_values("model_rank_position")["ticker"].tolist() == ["XLF", "XLB", "XLE", "XLC"]
    assert predictions.sort_values("realized_rank_position")["ticker"].tolist() == ["XLF", "XLE", "XLC", "XLB"]


def test_standard_and_cross_sectional_configs_write_separate_results():
    standard = json.loads(open("configs/mvp.yaml", encoding="utf-8").read())
    cross = json.loads(open("configs/cross_sectional_features.yaml", encoding="utf-8").read())
    assert standard["feature_selection"]["mode"] == "standard"
    assert cross["feature_selection"]["mode"] == "cross_sectional_aware"
    assert standard["results_dir"] != cross["results_dir"]


def test_end_to_end_cli_smoke(tmp_path):
    dataset = tmp_path / "data" / "qfactor_dataset.csv"
    results = tmp_path / "results"
    config = tmp_path / "mvp.yaml"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "qfactor_penny.prepare_data",
            "--input",
            str(tmp_path / "missing.csv"),
            "--output",
            str(dataset),
        ],
        check=True,
    )
    config.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset),
                "results_dir": str(results),
                "min_train_dates": 8,
                "validation_dates": 3,
                "purge_trading_days": 5,
                "max_splits": 1,
                "feature_count": 4,
                "qnn_epochs": 1,
                "qnn_learning_rate": 0.01,
                "qnn_patience": 1,
                "qnn_shot_sensitivity_samples": 4,
                "qnn_shot_sensitivity_shots": 1024,
                "mlp_hidden_units": 4,
                "transaction_cost": 0.0005,
                "random_state": 42,
                "seeds": [42, 7],
                "feature_selection": {"mode": "standard"},
            }
        ),
        encoding="utf-8",
    )
    subprocess.run([sys.executable, "-m", "qfactor_penny.run_benchmark", "--config", str(config)], check=True)
    subprocess.run([sys.executable, "-m", "qfactor_penny.make_report", "--config", str(config)], check=True)
    for name in [
        "metrics_summary.csv",
        "portfolio_summary.csv",
        "quantum_diagnostics.csv",
        "rebalance_predictions.csv",
        "split_audit.csv",
        "prediction_audit.csv",
        "model_aggregate_summary.csv",
        "portfolio_aggregate_summary.csv",
        "undefined_metric_audit.csv",
        "seed_stability_summary.csv",
        "path_dependence_summary.csv",
        "qnn_failure_audit.csv",
        "portfolio_selection_audit.csv",
        "feature_stability_summary.csv",
        "model_run_status.csv",
        "run_manifest.json",
        "experimental_variant_comparison.csv",
        "results_summary.md",
    ]:
        assert (results / name).exists()
    metrics = pd.read_csv(results / "metrics_summary.csv")
    assert set(metrics["seed"]) == {42, 7}
    assert metrics["model"].nunique() >= 4
    assert "pennylane_qnn" in set(metrics["model"])
    predictions = pd.read_csv(results / "rebalance_predictions.csv")
    assert set(predictions["seed"]) == {42, 7}
    assert predictions.groupby(["model", "date", "seed"])["ticker"].nunique().min() == 11
    assert predictions["label"].isna().any()
    assert {"model_rank_position", "realized_rank_position"}.issubset(predictions.columns)
    assert "rank_position" not in predictions.columns
    for _, group in predictions.groupby(["split_id", "date", "model", "seed"]):
        score_order = group.sort_values(["score", "ticker"], ascending=[False, True])["ticker"].tolist()
        rank_order = group.sort_values(["model_rank_position", "ticker"], ascending=[True, True])["ticker"].tolist()
        assert rank_order == score_order
    audit = pd.read_csv(results / "prediction_audit.csv")
    assert set(audit["seed"]) == {42, 7}
    assert audit["model_rank_matches_score"].all()
    assert audit["passes_shape_check"].all()
    portfolio_audit = pd.read_csv(results / "portfolio_selection_audit.csv")
    assert portfolio_audit["selection_matches_score_rank"].all()
    model_status = pd.read_csv(results / "model_run_status.csv")
    assert {"split_id", "seed", "model", "status", "num_predictions", "num_constant_score_dates"}.issubset(model_status.columns)
    assert set(model_status["seed"]) == {42, 7}
    assert {"naive_momentum", "pennylane_qnn"}.issubset(set(model_status["model"]))
    for _, group in model_status.groupby(["split_id", "seed"]):
        assert {"naive_momentum", "pennylane_qnn"}.issubset(set(group["model"]))
    feature_stability = pd.read_csv(results / "feature_stability_summary.csv")
    assert {"selected_feature", "selected_for_model", "is_calendar_heavy"}.issubset(feature_stability.columns)
    manifest = json.loads((results / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_hash"]
    assert manifest["dataset_hash"]
    assert manifest["feature_selection"]["mode"] == "standard"
    diagnostics = pd.read_csv(results / "quantum_diagnostics.csv")
    assert set(diagnostics["seed"]) == {42, 7}
    assert {"n_qubits", "n_layers", "shot_score_correlation", "shot_mean_abs_score_diff", "shot_ranking_flip_rate"}.issubset(
        diagnostics.columns
    )
    for figure in [
        "model_rank_ic.png",
        "portfolio_equity.png",
        "roc_auc_by_model.png",
        "alpha_vs_spy_by_model.png",
        "turnover_vs_return.png",
        "qnn_shot_sensitivity.png",
        "split_rank_ic_by_model.png",
    ]:
        assert (results / "figures" / figure).exists()
    summary = (results / "results_summary.md").read_text(encoding="utf-8")
    assert "It does not claim quantum advantage." in summary
    assert "/Users/" not in summary
    assert str(tmp_path) not in summary
    assert "Undefined Metric Audit" in summary
    assert "Portfolio Selection Audit" in summary
    assert "Seed Stability Audit" in summary
    assert "Path Dependence Audit" in summary
    assert "QNN Failure Audit" in summary
    assert "Model Aggregate Summary" in summary
    assert "alpha discovery" not in summary.lower()
    comparison = pd.read_csv(results / "experimental_variant_comparison.csv")
    assert "results_full" not in set(comparison["results_dir"])
    assert "unknown" not in set(comparison["feature_selection_mode"])
    assert comparison["selected_feature_rows"].gt(0).all()


def test_variant_comparison_excludes_stale_full_results(tmp_path):
    def write_current_results(name: str, mode: str) -> None:
        results_dir = tmp_path / name
        results_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "model": "pennylane_qnn",
                    "rank_ic": -0.1,
                    "precision_at_3": 0.25,
                    "roc_auc": 0.5,
                    "balanced_accuracy": 0.5,
                    "f1": 0.5,
                }
            ]
        ).to_csv(results_dir / "metrics_summary.csv", index=False)
        pd.DataFrame(
            [{"model": "pennylane_qnn", "net_return": 0.01, "alpha_vs_spy": -0.001, "turnover": 0.5}]
        ).to_csv(results_dir / "portfolio_summary.csv", index=False)
        (results_dir / "run_manifest.json").write_text(
            json.dumps({"feature_selection": {"mode": mode}, "max_splits": 4, "random_seeds": [42]}),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "split_id": "split_00",
                    "seed": 42,
                    "model": "pennylane_qnn",
                    "status": "success",
                    "num_constant_score_dates": 0,
                    "diagnostics_available": True,
                }
            ]
        ).to_csv(results_dir / "model_run_status.csv", index=False)
        pd.DataFrame(
            [
                {
                    "split_id": "split_00",
                    "seed": 42,
                    "selected_feature": "ret_1d",
                    "feature_selection_mode": mode,
                    "is_calendar_heavy": False,
                }
            ]
        ).to_csv(results_dir / "feature_stability_summary.csv", index=False)
        pd.DataFrame([{"constant_score_groups": 0, "any_undefined_metric": False}]).to_csv(
            results_dir / "qnn_failure_audit.csv", index=False
        )

    write_current_results("results", "standard")
    write_current_results("results_cross_sectional_mvp", "cross_sectional_aware")
    stale = tmp_path / "results_full"
    stale.mkdir()
    pd.DataFrame(
        [{"model": "pennylane_qnn", "rank_ic": 0.2, "precision_at_3": 0.5, "roc_auc": 0.5, "balanced_accuracy": 0.5, "f1": 0.5}]
    ).to_csv(stale / "metrics_summary.csv", index=False)
    pd.DataFrame([{"model": "pennylane_qnn", "net_return": 0.02, "alpha_vs_spy": 0.01, "turnover": 0.4}]).to_csv(
        stale / "portfolio_summary.csv", index=False
    )

    comparison = _variant_comparison(tmp_path, tmp_path / "results")
    assert set(comparison["results_dir"]) == {"results", "results_cross_sectional_mvp"}
    assert "results_full" not in set(comparison["results_dir"])
    assert "unknown" not in set(comparison["feature_selection_mode"])
    assert comparison["max_splits"].notna().all()
    assert comparison["selected_feature_rows"].gt(0).all()
