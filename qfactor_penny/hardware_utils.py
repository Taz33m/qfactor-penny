"""Shared helpers for optional IBM Quantum hardware audit CLIs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def load_env_file(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE lines without adding a runtime dependency."""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_frozen_qnn_subset(path: str | Path) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(artifact.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported frozen QNN hardware artifact schema.")
    return artifact


def flattened_samples(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in artifact.get("qnn_models", []):
        for sample in model.get("samples", []):
            row = {
                "model_id": model["model_id"],
                "split_id": model["split_id"],
                "seed": int(model["seed"]),
                "n_qubits": int(model["n_qubits"]),
                "n_layers": int(model["n_layers"]),
                "trainable_parameter_count": int(model["trainable_parameter_count"]),
                "selected_features": ",".join(model.get("selected_features", [])),
                "weights": model["weights"],
                "bias": float(model["bias"]),
                **sample,
            }
            rows.append(row)
    return rows


def scrub_error_message(message: str) -> str:
    token = os.environ.get("QISKIT_IBM_TOKEN")
    if token:
        message = message.replace(token, "<redacted>")
    return message.replace(str(Path.home()), "~")


def qiskit_runtime_available() -> bool:
    try:
        import qiskit_ibm_runtime  # noqa: F401
    except Exception:
        return False
    return True


def qiskit_available() -> bool:
    try:
        import qiskit  # noqa: F401
    except Exception:
        return False
    return True


def ibm_credentials_available() -> bool:
    return bool(os.environ.get("QISKIT_IBM_TOKEN"))


def skipped_rows(
    artifact: dict[str, Any],
    *,
    status: str,
    error_type: str,
    error_message: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for sample in flattened_samples(artifact):
        row = _public_sample_columns(sample)
        row.update(
            {
                "status": status,
                "error_type": error_type,
                "error_message": scrub_error_message(error_message),
            }
        )
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def write_rows(rows: Iterable[dict[str, Any]], output: str | Path) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def pairwise_flip_rate(reference: np.ndarray, candidate: np.ndarray) -> float:
    total = 0
    flipped = 0
    for left in range(len(reference)):
        for right in range(left + 1, len(reference)):
            ref_order = np.sign(reference[left] - reference[right])
            cand_order = np.sign(candidate[left] - candidate[right])
            if ref_order == 0 or cand_order == 0:
                continue
            total += 1
            flipped += int(ref_order != cand_order)
    return float(flipped / total) if total else float("nan")


def top3_overlap(reference: pd.Series, candidate: pd.Series, tickers: pd.Series) -> float:
    if len(reference) < 3 or len(candidate) < 3:
        return float("nan")
    frame = pd.DataFrame({"ticker": tickers, "reference": reference, "candidate": candidate})
    ref_top = set(frame.sort_values(["reference", "ticker"], ascending=[False, True]).head(3)["ticker"])
    cand_top = set(frame.sort_values(["candidate", "ticker"], ascending=[False, True]).head(3)["ticker"])
    return float(len(ref_top & cand_top) / 3.0)


def _public_sample_columns(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": sample.get("model_id"),
        "split_id": sample.get("split_id"),
        "seed": sample.get("seed"),
        "date": sample.get("date"),
        "ticker": sample.get("ticker"),
        "sample_index": sample.get("sample_index"),
        "analytic_score": sample.get("analytic_score"),
    }
