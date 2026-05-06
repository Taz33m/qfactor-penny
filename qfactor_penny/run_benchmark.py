"""Run QFactor-Penny benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config, project_path
from .constants import BENCHMARK_TICKER, SECTOR_TICKERS
from .metrics import evaluate_model_predictions
from .models import fit_classical_models
from .portfolio import aggregate_portfolio_metrics, build_portfolio_summary
from .preprocessing import FeaturePreprocessor
from .qnn import fit_predict_qnn
from .splits import make_walk_forward_splits


def _ensure_results(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = results_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def _slice(frame: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.DataFrame:
    return frame[frame["date"].isin(dates)].copy()


def _labeled(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["label"].notna()].copy()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(config: dict[str, Any], config_path: str | Path | None) -> str:
    if config_path is not None and Path(config_path).exists():
        return _hash_file(Path(config_path))
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "PennyLane": "PennyLane",
        "torch": "torch",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
    }
    versions: dict[str, str | None] = {}
    for label, package in packages.items():
        try:
            versions[label] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def _git_sha(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    sha = result.stdout.strip()
    return sha or None


def _score_diagnostics(predictions: pd.DataFrame | None) -> dict[str, int]:
    if predictions is None or predictions.empty:
        return {"num_predictions": 0, "num_constant_score_dates": 0, "num_nan_scores": 0}
    unique_by_date = predictions.groupby("date")["score"].nunique(dropna=True)
    return {
        "num_predictions": int(len(predictions)),
        "num_constant_score_dates": int(unique_by_date.le(1).sum()),
        "num_nan_scores": int(predictions["score"].isna().sum()),
    }


def _status_row(
    *,
    split_id: str,
    seed: int,
    model: str,
    status: str,
    fit_success: bool,
    predict_success: bool,
    error_type: str = "",
    error_message: str = "",
    num_train_rows: int,
    num_validation_rows: int,
    num_test_rows: int,
    predictions: pd.DataFrame | None = None,
    runtime_seconds: float = 0.0,
    diagnostics_available: bool = False,
) -> dict[str, object]:
    diagnostics = _score_diagnostics(predictions)
    return {
        "split_id": split_id,
        "seed": int(seed),
        "model": model,
        "status": status,
        "fit_success": bool(fit_success),
        "predict_success": bool(predict_success),
        "error_type": error_type,
        "error_message": error_message,
        "num_train_rows": int(num_train_rows),
        "num_validation_rows": int(num_validation_rows),
        "num_test_rows": int(num_test_rows),
        "num_predictions": diagnostics["num_predictions"],
        "num_constant_score_dates": diagnostics["num_constant_score_dates"],
        "num_nan_scores": diagnostics["num_nan_scores"],
        "runtime_seconds": float(runtime_seconds),
        "diagnostics_available": bool(diagnostics_available),
    }


def _business_feature_stats(train_frame: pd.DataFrame, feature: str) -> dict[str, object]:
    series = train_frame[feature]
    train_std = float(series.std(ddof=0))
    cross_sectional = train_frame.groupby("date")[feature].std(ddof=0)
    time_series = train_frame.groupby("ticker")[feature].std(ddof=0)
    mean_cross = float(cross_sectional.mean()) if len(cross_sectional) else math.nan
    mean_time = float(time_series.mean()) if len(time_series) else math.nan
    ratio = (mean_cross**2) / (mean_time**2) if mean_time and np.isfinite(mean_time) else math.nan
    return {
        "train_mean": float(series.mean()),
        "train_std": train_std,
        "mean_cross_sectional_std_by_date": mean_cross,
        "mean_time_series_std_by_ticker": mean_time,
        "cross_sectional_to_time_series_variance_ratio": ratio,
        "missing_rate": float(series.isna().mean()),
        "is_calendar_heavy": bool(np.isfinite(ratio) and ratio < 0.10),
    }


def _feature_stability_rows(
    *,
    split_id: str,
    seed: int,
    train_frame: pd.DataFrame,
    selected_features: list[str],
    selected_for_model: str,
    feature_selection_mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in selected_features:
        rows.append(
            {
                "split_id": split_id,
                "seed": int(seed),
                "selected_feature": feature,
                "selected_for_model": selected_for_model,
                "feature_selection_mode": feature_selection_mode,
                **_business_feature_stats(train_frame, feature),
            }
        )
    return rows


def _write_run_manifest(
    *,
    results_dir: Path,
    config: dict[str, Any],
    config_path: str | Path | None,
    dataset_path: Path,
    frame: pd.DataFrame,
    model_list: list[str],
    seeds: list[int],
    run_start: datetime,
    run_end: datetime,
    root: Path,
) -> None:
    manifest = {
        "config_path": str(config_path) if config_path is not None else "<in-memory>",
        "config_hash": _config_hash(config, config_path),
        "config": config,
        "max_splits": config.get("max_splits"),
        "feature_selection": config.get("feature_selection", {"mode": "standard"}),
        "dataset_path": str(dataset_path),
        "dataset_hash": _hash_file(dataset_path),
        "synthetic_flag": bool(frame["is_synthetic"].astype(bool).any()) if "is_synthetic" in frame else None,
        "prepared_row_count": int(len(frame)),
        "rebalance_date_count": int(frame["date"].nunique()),
        "model_list": sorted(model_list),
        "random_seeds": seeds,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "dependency_versions": _dependency_versions(),
        "git_sha": _git_sha(root),
        "run_start_timestamp": run_start.isoformat(),
        "run_end_timestamp": run_end.isoformat(),
        "runtime_seconds": (run_end - run_start).total_seconds(),
        "output_directory": str(results_dir),
    }
    (results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _prediction_rows(
    *,
    split_id: str,
    model: str,
    test_frame: pd.DataFrame,
    scores: np.ndarray,
    selected_features: list[str],
    train_seconds: float,
    inference_seconds: float,
    param_count: int | float,
    seed: int,
) -> pd.DataFrame:
    out = test_frame.copy()
    if "realized_rank_position" not in out.columns:
        out["realized_rank_position"] = math.nan
    out["split_id"] = split_id
    out["model"] = model
    out["score"] = np.round(np.asarray(scores, dtype=float), 12)
    ranked = out.sort_values(["date", "score", "ticker"], ascending=[True, False, True])
    out["model_rank_position"] = ranked.groupby("date").cumcount().add(1).reindex(out.index).astype(int)
    out["selected_features"] = ",".join(selected_features)
    out["train_seconds"] = float(train_seconds)
    out["inference_seconds"] = float(inference_seconds)
    out["param_count"] = param_count
    out["seed"] = int(seed)
    out["benchmark_ticker"] = BENCHMARK_TICKER
    return out[
        [
            "split_id",
            "seed",
            "date",
            "ticker",
            "model",
            "score",
            "label",
            "model_rank_position",
            "realized_rank_position",
            "forward_return_5d",
            "spy_forward_return_5d",
            "excess_return_5d",
            "selected_features",
            "train_seconds",
            "inference_seconds",
            "param_count",
            "is_synthetic",
            "data_source",
            "benchmark_ticker",
        ]
    ]


def _naive_momentum_scores(test_frame: pd.DataFrame) -> np.ndarray:
    if "relative_strength_20d" in test_frame:
        return test_frame["relative_strength_20d"].to_numpy(dtype=float)
    return test_frame["ret_20d"].to_numpy(dtype=float)


def _write_outputs(
    *,
    results_dir: Path,
    metrics_rows: list[dict[str, Any]],
    predictions: pd.DataFrame,
    split_audit: pd.DataFrame,
    quantum_rows: list[dict[str, Any]],
    model_status_rows: list[dict[str, Any]],
    feature_stability_rows: list[dict[str, Any]],
    transaction_cost: float,
) -> None:
    portfolio = build_portfolio_summary(predictions, transaction_cost=transaction_cost)
    portfolio_aggregates = aggregate_portfolio_metrics(portfolio)
    metrics = pd.DataFrame(metrics_rows)
    for column in [
        "portfolio_net_return_mean",
        "portfolio_alpha_mean",
        "portfolio_sharpe",
        "portfolio_max_drawdown",
        "portfolio_turnover_mean",
    ]:
        metrics[column] = metrics.apply(
            lambda row: portfolio_aggregates.get((row["model"], int(row["seed"])), {}).get(column, math.nan),
            axis=1,
        )
    metrics.to_csv(results_dir / "metrics_summary.csv", index=False)
    portfolio.to_csv(results_dir / "portfolio_summary.csv", index=False)
    pd.DataFrame(quantum_rows).to_csv(results_dir / "quantum_diagnostics.csv", index=False)
    pd.DataFrame(model_status_rows).to_csv(results_dir / "model_run_status.csv", index=False)
    pd.DataFrame(feature_stability_rows).to_csv(results_dir / "feature_stability_summary.csv", index=False)
    predictions.to_csv(results_dir / "rebalance_predictions.csv", index=False)
    split_audit.to_csv(results_dir / "split_audit.csv", index=False)
    aggregate_metrics(metrics).to_csv(results_dir / "model_aggregate_summary.csv", index=False)
    aggregate_portfolio(portfolio).to_csv(results_dir / "portfolio_aggregate_summary.csv", index=False)


def _prediction_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split_id, date, model, seed), group in predictions.groupby(["split_id", "date", "model", "seed"]):
        num_scores = int(group["score"].notna().sum())
        num_sector_etfs = int(group["ticker"].nunique())
        num_labeled = int(group["label"].notna().sum())
        num_middle = int(group["label"].isna().sum())
        score_ranked_tickers = group.sort_values(["score", "ticker"], ascending=[False, True])["ticker"].tolist()
        position_ranked_tickers = group.sort_values(["model_rank_position", "ticker"], ascending=[True, True])["ticker"].tolist()
        rank_matches_score = score_ranked_tickers == position_ranked_tickers
        rows.append(
            {
                "split_id": split_id,
                "date": date,
                "model": model,
                "seed": int(seed),
                "num_scores": num_scores,
                "num_sector_etfs": num_sector_etfs,
                "num_labeled": num_labeled,
                "num_middle_included": num_middle,
                "all_sector_scores": bool(group["score"].notna().all()),
                "model_rank_matches_score": bool(rank_matches_score),
                "passes_shape_check": bool(
                    num_sector_etfs == len(SECTOR_TICKERS)
                    and num_labeled == 6
                    and num_middle == 5
                    and group["score"].notna().all()
                    and rank_matches_score
                ),
            }
        )
    return pd.DataFrame(rows)


def _mean_std(frame: pd.DataFrame, *, group_by: list[str], columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    if frame.empty or not available:
        return pd.DataFrame()
    grouped = frame.groupby(group_by, as_index=False)[available].agg(["mean", "std"])
    grouped.columns = [
        "_".join([part for part in column if part]) if isinstance(column, tuple) else str(column)
        for column in grouped.columns.to_flat_index()
    ]
    return grouped


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return _mean_std(
        metrics,
        group_by=["model"],
        columns=[
            "roc_auc",
            "balanced_accuracy",
            "f1",
            "precision_at_3",
            "rank_ic",
            "portfolio_net_return_mean",
            "portfolio_alpha_mean",
            "portfolio_sharpe",
            "portfolio_max_drawdown",
            "portfolio_turnover_mean",
        ],
    )


def aggregate_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    return _mean_std(
        portfolio,
        group_by=["model"],
        columns=["gross_return", "net_return", "alpha_vs_spy", "turnover", "transaction_cost"],
    )


def _config_seeds(config: dict[str, Any]) -> list[int]:
    raw = config.get("seeds")
    if raw is None:
        return [int(config.get("random_state", 42))]
    if isinstance(raw, int):
        return [int(raw)]
    seeds = [int(seed) for seed in raw]
    if not seeds:
        raise ValueError("Config `seeds` must contain at least one seed.")
    return seeds


def run_benchmark(
    config: dict[str, Any],
    *,
    root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, int]:
    run_start = datetime.now(timezone.utc)
    base = Path.cwd() if root is None else Path(root)
    dataset_path = project_path(config["dataset_path"], root=base)
    results_dir = project_path(config.get("results_dir", "results"), root=base)
    _ensure_results(results_dir)

    frame = pd.read_csv(dataset_path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["forward_end_date"] = pd.to_datetime(frame["forward_end_date"])
    splits, split_audit = make_walk_forward_splits(
        frame,
        min_train_dates=int(config.get("min_train_dates", 36)),
        validation_dates=int(config.get("validation_dates", 6)),
        purge_trading_days=int(config.get("purge_trading_days", 5)),
        max_splits=config.get("max_splits"),
    )
    if not splits:
        raise SystemExit("No valid walk-forward splits were created.")

    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    quantum_rows: list[dict[str, Any]] = []
    model_status_rows: list[dict[str, Any]] = []
    feature_stability: list[dict[str, Any]] = []
    seeds = _config_seeds(config)
    feature_selection = config.get("feature_selection", {}) or {}
    feature_selection_mode = str(feature_selection.get("mode", "standard"))
    min_cross_sectional_quantile = float(feature_selection.get("min_cross_sectional_std_quantile", 0.25))

    for seed in seeds:
        for split in splits:
            train_frame = _slice(frame, split.train_dates)
            validation_frame = _slice(frame, split.validation_dates)
            test_frame = _slice(frame, split.test_dates)
            train_labeled = _labeled(train_frame)
            validation_labeled = _labeled(validation_frame)
            if train_labeled.empty or validation_labeled.empty:
                warnings.warn(f"{split.split_id} lacks labeled train/validation rows; skipping.", RuntimeWarning)
                continue

            preprocessor = FeaturePreprocessor(
                feature_count=int(config.get("feature_count", 4)),
                random_state=seed,
                feature_selection_mode=feature_selection_mode,
                min_cross_sectional_std_quantile=min_cross_sectional_quantile,
            ).fit(
                train_labeled,
                train_labeled["label"].to_numpy(dtype=int),
            )
            x_train = preprocessor.transform(train_labeled)
            y_train = train_labeled["label"].to_numpy(dtype=int)
            x_validation = preprocessor.transform(validation_labeled)
            y_validation = validation_labeled["label"].to_numpy(dtype=int)
            x_test = preprocessor.transform(test_frame)
            learned_features = preprocessor.selected_features or []
            feature_stability.extend(
                _feature_stability_rows(
                    split_id=split.split_id,
                    seed=seed,
                    train_frame=train_frame,
                    selected_features=learned_features,
                    selected_for_model="learned_models",
                    feature_selection_mode=feature_selection_mode,
                )
            )
            feature_stability.extend(
                _feature_stability_rows(
                    split_id=split.split_id,
                    seed=seed,
                    train_frame=train_frame,
                    selected_features=["relative_strength_20d"],
                    selected_for_model="naive_momentum",
                    feature_selection_mode="fixed_naive",
                )
            )

            naive_start = time.perf_counter()
            naive_scores = _naive_momentum_scores(test_frame)
            naive_predictions = _prediction_rows(
                split_id=split.split_id,
                model="naive_momentum",
                test_frame=test_frame,
                scores=naive_scores,
                selected_features=["relative_strength_20d"],
                train_seconds=0.0,
                inference_seconds=0.0,
                param_count=0,
                seed=seed,
            )
            prediction_frames.append(naive_predictions)
            metrics_rows.append(evaluate_model_predictions(naive_predictions, model="naive_momentum", split_id=split.split_id))
            model_status_rows.append(
                _status_row(
                    split_id=split.split_id,
                    seed=seed,
                    model="naive_momentum",
                    status="success",
                    fit_success=True,
                    predict_success=True,
                    num_train_rows=len(train_labeled),
                    num_validation_rows=len(validation_labeled),
                    num_test_rows=len(test_frame),
                    predictions=naive_predictions,
                    runtime_seconds=time.perf_counter() - naive_start,
                    diagnostics_available=False,
                )
            )

            trained_models, fit_statuses = fit_classical_models(
                x_train,
                y_train,
                random_state=seed,
                mlp_hidden_units=int(config.get("mlp_hidden_units", 4)),
                return_status=True,
            )
            trained_names = {model.name for model in trained_models}
            for fit_status in fit_statuses:
                name = str(fit_status["model"])
                if name in trained_names:
                    continue
                model_status_rows.append(
                    _status_row(
                        split_id=split.split_id,
                        seed=seed,
                        model=name,
                        status=str(fit_status["status"]),
                        fit_success=bool(fit_status["fit_success"]),
                        predict_success=False,
                        error_type=str(fit_status["error_type"]),
                        error_message=str(fit_status["error_message"]),
                        num_train_rows=len(train_labeled),
                        num_validation_rows=len(validation_labeled),
                        num_test_rows=len(test_frame),
                        predictions=None,
                        runtime_seconds=float(fit_status["train_seconds"]),
                        diagnostics_available=False,
                    )
                )
            for model in trained_models:
                infer_start = time.perf_counter()
                try:
                    scores = model.score(x_test)
                    inference_seconds = time.perf_counter() - infer_start
                    predictions = _prediction_rows(
                        split_id=split.split_id,
                        model=model.name,
                        test_frame=test_frame,
                        scores=scores,
                        selected_features=learned_features,
                        train_seconds=model.train_seconds,
                        inference_seconds=inference_seconds,
                        param_count=model.param_count,
                        seed=seed,
                    )
                    prediction_frames.append(predictions)
                    metrics_rows.append(evaluate_model_predictions(predictions, model=model.name, split_id=split.split_id))
                    model_status_rows.append(
                        _status_row(
                            split_id=split.split_id,
                            seed=seed,
                            model=model.name,
                            status="success",
                            fit_success=True,
                            predict_success=True,
                            num_train_rows=len(train_labeled),
                            num_validation_rows=len(validation_labeled),
                            num_test_rows=len(test_frame),
                            predictions=predictions,
                            runtime_seconds=model.train_seconds + inference_seconds,
                            diagnostics_available=False,
                        )
                    )
                except Exception as exc:
                    model_status_rows.append(
                        _status_row(
                            split_id=split.split_id,
                            seed=seed,
                            model=model.name,
                            status="failed",
                            fit_success=True,
                            predict_success=False,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            num_train_rows=len(train_labeled),
                            num_validation_rows=len(validation_labeled),
                            num_test_rows=len(test_frame),
                            predictions=None,
                            runtime_seconds=model.train_seconds + (time.perf_counter() - infer_start),
                            diagnostics_available=False,
                        )
                    )

            qnn_start = time.perf_counter()
            try:
                qnn = fit_predict_qnn(
                    x_train,
                    y_train,
                    x_validation,
                    y_validation,
                    x_test,
                    selected_features=learned_features,
                    epochs=int(config.get("qnn_epochs", 10)),
                    learning_rate=float(config.get("qnn_learning_rate", 0.08)),
                    patience=int(config.get("qnn_patience", 3)),
                    random_state=seed,
                    shot_sensitivity_samples=int(config.get("qnn_shot_sensitivity_samples", 8)),
                    shot_sensitivity_shots=int(config.get("qnn_shot_sensitivity_shots", 1024)),
                )
                qnn_predictions = _prediction_rows(
                    split_id=split.split_id,
                    model="pennylane_qnn",
                    test_frame=test_frame,
                    scores=qnn.score,
                    selected_features=qnn.selected_features,
                    train_seconds=qnn.train_seconds,
                    inference_seconds=qnn.inference_seconds,
                    param_count=qnn.parameter_count,
                    seed=seed,
                )
                prediction_frames.append(qnn_predictions)
                metrics_rows.append(evaluate_model_predictions(qnn_predictions, model="pennylane_qnn", split_id=split.split_id))
                quantum_rows.append(
                    {
                        "split_id": split.split_id,
                        "seed": seed,
                        "model": "pennylane_qnn",
                        "n_qubits": qnn.qubits,
                        "n_layers": qnn.layers,
                        "trainable_parameter_count": qnn.parameter_count,
                        "train_seconds": qnn.train_seconds,
                        "inference_seconds": qnn.inference_seconds,
                        "simulation_mode": "analytic_default_qubit",
                        "shots": "analytic",
                        "shot_sensitivity_samples": qnn.shot_sensitivity_samples,
                        "shot_sensitivity_shots": qnn.shot_sensitivity_shots,
                        "shot_score_correlation": qnn.shot_score_correlation,
                        "shot_mean_abs_score_diff": qnn.shot_mean_abs_score_diff,
                        "shot_ranking_flip_rate": qnn.shot_ranking_flip_rate,
                        "epochs_ran": qnn.epochs_ran,
                        "best_validation_loss": qnn.best_validation_loss,
                        "selected_features": ",".join(qnn.selected_features),
                    }
                )
                model_status_rows.append(
                    _status_row(
                        split_id=split.split_id,
                        seed=seed,
                        model="pennylane_qnn",
                        status="success",
                        fit_success=True,
                        predict_success=True,
                        num_train_rows=len(train_labeled),
                        num_validation_rows=len(validation_labeled),
                        num_test_rows=len(test_frame),
                        predictions=qnn_predictions,
                        runtime_seconds=time.perf_counter() - qnn_start,
                        diagnostics_available=True,
                    )
                )
            except Exception as exc:
                warnings.warn(f"PennyLane QNN failed for {split.split_id} seed {seed}: {exc}", RuntimeWarning)
                model_status_rows.append(
                    _status_row(
                        split_id=split.split_id,
                        seed=seed,
                        model="pennylane_qnn",
                        status="failed",
                        fit_success=False,
                        predict_success=False,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        num_train_rows=len(train_labeled),
                        num_validation_rows=len(validation_labeled),
                        num_test_rows=len(test_frame),
                        predictions=None,
                        runtime_seconds=time.perf_counter() - qnn_start,
                        diagnostics_available=False,
                    )
                )

    if not prediction_frames:
        raise SystemExit("No model predictions were produced.")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    _write_outputs(
        results_dir=results_dir,
        metrics_rows=metrics_rows,
        predictions=predictions,
        split_audit=split_audit,
        quantum_rows=quantum_rows,
        model_status_rows=model_status_rows,
        feature_stability_rows=feature_stability,
        transaction_cost=float(config.get("transaction_cost", 0.0005)),
    )
    _prediction_audit(predictions).to_csv(results_dir / "prediction_audit.csv", index=False)
    run_end = datetime.now(timezone.utc)
    _write_run_manifest(
        results_dir=results_dir,
        config=config,
        config_path=config_path,
        dataset_path=dataset_path,
        frame=frame,
        model_list=sorted({str(row["model"]) for row in model_status_rows}),
        seeds=seeds,
        run_start=run_start,
        run_end=run_end,
        root=base,
    )
    return {
        "splits": len(splits),
        "models": int(predictions["model"].nunique()),
        "prediction_rows": int(len(predictions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QFactor-Penny benchmark.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_benchmark(load_config(args.config), config_path=args.config))


if __name__ == "__main__":
    main()
