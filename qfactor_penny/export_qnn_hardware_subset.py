"""Export frozen QNN circuits for optional IBM Quantum hardware audits."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config, project_path
from .preprocessing import FeaturePreprocessor
from .qnn import fit_predict_qnn
from .run_benchmark import _config_hash, _hash_file, _labeled, _slice, _config_seeds
from .splits import make_walk_forward_splits


def export_qnn_hardware_subset(
    config: dict[str, Any],
    *,
    output: str | Path,
    root: str | Path | None = None,
    config_path: str | Path | None = None,
    max_models: int = 1,
    max_dates_per_model: int = 1,
    split_id: str | None = None,
    seed_filter: int | None = None,
) -> dict[str, Any]:
    base = Path.cwd() if root is None else Path(root)
    output_path = Path(output)
    cache_dir = output_path.parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    dataset_path = project_path(config["dataset_path"], root=base)
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
    if split_id is not None:
        splits = [split for split in splits if split.split_id == split_id]
    if not splits:
        raise SystemExit("No matching walk-forward splits were available for QNN hardware export.")

    feature_selection = config.get("feature_selection", {}) or {}
    feature_selection_mode = str(feature_selection.get("mode", "standard"))
    min_cross_sectional_quantile = float(feature_selection.get("min_cross_sectional_std_quantile", 0.25))
    seeds = _config_seeds(config)
    if seed_filter is not None:
        seeds = [seed for seed in seeds if seed == seed_filter]
    if not seeds:
        raise SystemExit("No matching seed was available for QNN hardware export.")

    qnn_models: list[dict[str, Any]] = []
    for seed in seeds:
        for split in splits:
            if len(qnn_models) >= max_models:
                break
            train_frame = _slice(frame, split.train_dates)
            validation_frame = _slice(frame, split.validation_dates)
            test_frame = _slice(frame, split.test_dates).reset_index(drop=True)
            train_labeled = _labeled(train_frame)
            validation_labeled = _labeled(validation_frame)
            if train_labeled.empty or validation_labeled.empty:
                continue

            preprocessor = FeaturePreprocessor(
                feature_count=int(config.get("feature_count", 4)),
                random_state=seed,
                feature_selection_mode=feature_selection_mode,
                min_cross_sectional_std_quantile=min_cross_sectional_quantile,
            ).fit(train_labeled, train_labeled["label"].to_numpy(dtype=int))
            x_train = preprocessor.transform(train_labeled)
            y_train = train_labeled["label"].to_numpy(dtype=int)
            x_validation = preprocessor.transform(validation_labeled)
            y_validation = validation_labeled["label"].to_numpy(dtype=int)
            x_test = preprocessor.transform(test_frame)
            qnn = fit_predict_qnn(
                x_train,
                y_train,
                x_validation,
                y_validation,
                x_test,
                selected_features=preprocessor.selected_features or [],
                epochs=int(config.get("qnn_epochs", 10)),
                learning_rate=float(config.get("qnn_learning_rate", 0.08)),
                patience=int(config.get("qnn_patience", 3)),
                random_state=seed,
                shot_sensitivity_samples=0,
                shot_sensitivity_shots=0,
            )
            selected_dates = sorted(test_frame["date"].drop_duplicates())[: max(1, int(max_dates_per_model))]
            sample_mask = test_frame["date"].isin(selected_dates)
            sample_indices = [int(index) for index in np.flatnonzero(sample_mask.to_numpy())]
            samples = [_sample_row(test_frame.iloc[index], x_test[index], qnn.score[index], index) for index in sample_indices]
            if not samples:
                continue
            qnn_models.append(
                {
                    "model_id": f"{split.split_id}__seed_{seed}",
                    "split_id": split.split_id,
                    "seed": int(seed),
                    "selected_features": qnn.selected_features,
                    "weights": np.asarray(qnn.weights, dtype=float).tolist(),
                    "bias": float(qnn.bias),
                    "n_qubits": int(qnn.qubits),
                    "n_layers": int(qnn.layers),
                    "trainable_parameter_count": int(qnn.parameter_count),
                    "epochs_ran": int(qnn.epochs_ran),
                    "best_validation_loss": float(qnn.best_validation_loss),
                    "train_rows": int(len(train_labeled)),
                    "validation_rows": int(len(validation_labeled)),
                    "test_rows": int(len(test_frame)),
                    "samples": samples,
                }
            )
        if len(qnn_models) >= max_models:
            break

    if not qnn_models:
        raise SystemExit("No frozen QNN models were exported.")

    artifact = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "inference-only IBM Quantum hardware robustness audit",
        "claim_discipline": (
            "Hardware execution is a systems diagnostic only; it does not support quantum-advantage, "
            "trading-edge, or real-hardware validated finance-signal claims."
        ),
        "source_config_path": str(config_path) if config_path is not None else "<in-memory>",
        "config_hash": _config_hash(config, config_path),
        "dataset_path": str(config.get("dataset_path")),
        "dataset_hash": _hash_file(dataset_path),
        "feature_selection": feature_selection or {"mode": "standard"},
        "max_splits": config.get("max_splits"),
        "split_audit_rows": split_audit.to_dict(orient="records"),
        "qnn_architecture": {
            "framework": "PennyLane training, Qiskit hardware export",
            "feature_count": 4,
            "n_qubits": 4,
            "encoding": "RY angle encoding",
            "ansatz": "one StronglyEntanglingLayers layer",
            "observable": "PauliZ on qubit 0 plus classical bias",
            "training_mode": "analytic default.qubit",
            "hardware_mode": "inference only",
        },
        "qnn_models": qnn_models,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def _sample_row(row: pd.Series, feature_values: np.ndarray, analytic_score: float, index: int) -> dict[str, Any]:
    label = row.get("label")
    return {
        "sample_index": int(index),
        "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
        "ticker": str(row["ticker"]),
        "label": None if pd.isna(label) else float(label),
        "feature_values": np.asarray(feature_values, dtype=float).tolist(),
        "analytic_score": float(analytic_score),
        "forward_return_5d": float(row["forward_return_5d"]),
        "spy_forward_return_5d": float(row["spy_forward_return_5d"]),
        "excess_return_5d": float(row["excess_return_5d"]),
        "is_synthetic": bool(row.get("is_synthetic", False)),
        "data_source": str(row.get("data_source", "")),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Benchmark config used to train the frozen QNN subset.")
    parser.add_argument("--output", required=True, help="Output JSON path, e.g. results_hardware/frozen_qnn_subset.json.")
    parser.add_argument("--max-models", type=int, default=1, help="Maximum split/seed QNN models to export.")
    parser.add_argument("--max-dates-per-model", type=int, default=1, help="Full rebalance dates exported per QNN model.")
    parser.add_argument("--split-id", default=None, help="Optional split_id filter.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed filter.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    export_qnn_hardware_subset(
        config,
        output=args.output,
        config_path=args.config,
        max_models=args.max_models,
        max_dates_per_model=args.max_dates_per_model,
        split_id=args.split_id,
        seed_filter=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
