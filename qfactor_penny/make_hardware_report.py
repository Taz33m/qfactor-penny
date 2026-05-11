"""Create Markdown and manifest artifacts for optional IBM Quantum hardware audits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .hardware_utils import load_env_file, pairwise_flip_rate, top3_overlap


def make_hardware_report(input_dir: str | Path) -> Path:
    load_env_file()
    base = Path(input_dir)
    base.mkdir(parents=True, exist_ok=True)
    transpile = _read_optional(base / "ibm_transpilation_audit.csv")
    scores = _read_optional(base / "ibm_hardware_scores.csv")
    figure_paths = _write_figures(base, scores)
    manifest = _write_manifest(base, transpile, scores, figure_paths)
    lines = [
        "# IBM Quantum Hardware Audit",
        "",
        "This optional report treats IBM Quantum execution as an inference-time systems diagnostic. "
        "It does not claim quantum advantage, tradable alpha, or real-hardware validation of a finance signal.",
        "",
        "## Executive Interpretation",
        "",
        _executive_interpretation(transpile, scores),
        "",
        "## Artifacts",
        "",
        "| Artifact | Status |",
        "| --- | --- |",
        f"| `frozen_qnn_subset.json` | {_exists(base / 'frozen_qnn_subset.json')} |",
        f"| `ibm_transpilation_audit.csv` | {_exists(base / 'ibm_transpilation_audit.csv')} |",
        f"| `ibm_hardware_scores.csv` | {_exists(base / 'ibm_hardware_scores.csv')} |",
        f"| `hardware_run_manifest.json` | {_exists(base / 'hardware_run_manifest.json')} |",
        "",
        "## Transpilation Audit",
        "",
        _transpile_summary(transpile),
        "",
        "## Hardware Score Diagnostics",
        "",
        _score_summary(scores),
        "",
        "## HAL/Runtime Execution Note",
        "",
        _execution_note(scores, manifest),
        "",
        "## Diagnostic Figure",
        "",
        _figure_summary(base, figure_paths),
        "",
        "## Interpretation Guardrail",
        "",
        "A successful hardware run only shows how the frozen QNN circuits behave under a selected backend, "
        "transpilation pass, finite-shot estimator execution, and optional mitigation settings. It should be read "
        "alongside the existing negative simulator benchmark, not as a new performance result.",
        "",
    ]
    output = base / "hardware_summary.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _exists(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _transpile_summary(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No transpilation audit CSV is available yet."
    status_counts = frame["status"].value_counts(dropna=False).to_dict() if "status" in frame else {}
    lines = [f"Status counts: `{status_counts}`."]
    success = frame[frame.get("status", "") == "success"].copy()
    if success.empty:
        if "error_message" in frame:
            errors = frame[["error_type", "error_message"]].drop_duplicates().head(3)
            lines.append(_markdown_table(errors))
        return "\n\n".join(lines)
    summary = success.groupby("backend_name", dropna=False).agg(
        circuits=("ticker", "count"),
        pre_depth_mean=("pre_depth", "mean"),
        post_depth_mean=("post_depth", "mean"),
        post_two_qubit_gate_count_mean=("post_two_qubit_gate_count", "mean"),
        post_swap_count_mean=("post_swap_count", "mean"),
    )
    lines.append(_markdown_table(summary.reset_index()))
    return "\n\n".join(lines)


def _score_summary(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No IBM hardware score CSV is available yet."
    status_counts = frame["status"].value_counts(dropna=False).to_dict() if "status" in frame else {}
    lines = [f"Status counts: `{status_counts}`."]
    success = frame[frame.get("status", "") == "success"].copy()
    failed = frame[frame.get("status", "") != "success"].copy() if "status" in frame else pd.DataFrame()
    if success.empty:
        if "error_message" in frame:
            errors = frame[["error_type", "error_message"]].drop_duplicates().head(3)
            lines.append(_markdown_table(errors))
        return "\n\n".join(lines)
    lines.append(_markdown_table(pd.DataFrame(_score_diagnostic_rows(success))))
    top3 = pd.DataFrame(_top3_comparison(success))
    if not top3.empty:
        top3_display = top3.copy()
        top3_display["analytic_top3"] = top3_display["analytic_top3"].map(lambda values: ", ".join(values))
        top3_display["hardware_top3"] = top3_display["hardware_top3"].map(lambda values: ", ".join(values))
        lines.append("Top-3 ranking comparison:")
        lines.append(_markdown_table(top3_display))
    if not failed.empty and {"error_type", "error_message"}.issubset(failed.columns):
        errors = failed[["status", "error_type", "error_message"]].drop_duplicates().head(5)
        lines.append("Non-success hardware rows were also recorded:")
        lines.append(_markdown_table(errors))
    return "\n\n".join(lines)


def _executive_interpretation(transpile: pd.DataFrame, scores: pd.DataFrame) -> str:
    success_scores = scores[scores.get("status", "") == "success"].copy() if not scores.empty else pd.DataFrame()
    success_transpile = transpile[transpile.get("status", "") == "success"].copy() if not transpile.empty else pd.DataFrame()
    if success_scores.empty:
        return "No successful hardware inference rows are available yet."
    diagnostics = _score_diagnostic_rows(success_scores)
    row = diagnostics[0] if diagnostics else {}
    swap_text = ""
    if not success_transpile.empty and "post_swap_count" in success_transpile:
        swap_text = f" The transpiled circuits had mean SWAP count `{success_transpile['post_swap_count'].mean():.4f}`."
    return (
        "The frozen QNN circuits reached IBM hardware, but the real-device scores were unstable relative to "
        f"analytic simulation: score correlation `{row.get('score_correlation', math.nan):.4f}`, mean absolute "
        f"score difference `{row.get('mean_abs_score_diff', math.nan):.4f}`, pairwise ranking flip rate "
        f"`{row.get('pairwise_ranking_flip_rate', math.nan):.4f}`, and mean top-3 overlap "
        f"`{row.get('mean_top3_overlap', math.nan):.4f}`.{swap_text} This supports a hardware robustness "
        "reading, not a performance claim."
    )


def _execution_note(scores: pd.DataFrame, manifest: dict) -> str:
    rows = [note for note in manifest.get("execution_notes", []) if isinstance(note, dict)]
    if not scores.empty and "status" in scores:
        failed = scores[scores["status"] != "success"].copy()
        if not failed.empty and {"status", "error_type", "error_message"}.issubset(failed.columns):
            for column in ["backend_name", "optimization_level"]:
                if column not in failed.columns:
                    failed[column] = ""
            for row in failed[["status", "error_type", "error_message", "backend_name", "optimization_level"]].drop_duplicates().itertuples(index=False):
                rows.append(
                    {
                        "event": "recorded_non_success_row",
                        "status": row.status,
                        "backend_name": row.backend_name,
                        "optimization_level": row.optimization_level,
                        "error_type": row.error_type,
                        "error_code": "",
                        "interpretation": str(row.error_message)[:140],
                    }
                )
    if not rows:
        return (
            "The canonical CSV contains the successful conservative hardware run. Any failed exploratory attempts "
            "should be documented in `hardware_run_manifest.json` before regenerating this report."
        )
    columns = ["event", "status", "backend_name", "optimization_level", "error_type", "error_code", "interpretation"]
    return _markdown_table(pd.DataFrame(rows).reindex(columns=columns))


def _figure_summary(base: Path, figure_paths: list[Path]) -> str:
    if not figure_paths:
        return "No hardware diagnostic figure was generated."
    rows = []
    for path in figure_paths:
        try:
            rel = path.relative_to(base).as_posix()
        except ValueError:
            rel = path.name
        rows.append(f"- `{rel}`")
    return "\n".join(rows)


def _score_diagnostic_rows(success: pd.DataFrame) -> list[dict[str, object]]:
    required = {"backend_name", "resilience_level", "analytic_score", "hardware_score", "ticker", "model_id", "date"}
    if success.empty or not required.issubset(success.columns):
        return []
    rows: list[dict[str, object]] = []
    for (backend, resilience), group in success.groupby(["backend_name", "resilience_level"], dropna=False):
        analytic = group["analytic_score"].to_numpy(dtype=float)
        hardware = group["hardware_score"].to_numpy(dtype=float)
        corr = float(np.corrcoef(analytic, hardware)[0, 1]) if len(group) > 1 and len(set(analytic)) > 1 and len(set(hardware)) > 1 else math.nan
        top3_values = [
            top3_overlap(date_group["analytic_score"], date_group["hardware_score"], date_group["ticker"])
            for _, date_group in group.groupby(["model_id", "date"])
        ]
        rows.append(
            {
                "backend_name": backend,
                "resilience_level": resilience,
                "samples": int(len(group)),
                "score_correlation": corr,
                "mean_abs_score_diff": float(np.mean(np.abs(analytic - hardware))),
                "pairwise_ranking_flip_rate": pairwise_flip_rate(analytic, hardware),
                "mean_top3_overlap": float(np.nanmean(top3_values)) if top3_values else math.nan,
            }
        )
    return rows


def _write_figures(base: Path, scores: pd.DataFrame) -> list[Path]:
    success = scores[scores.get("status", "") == "success"].copy() if not scores.empty else pd.DataFrame()
    if success.empty or not {"analytic_score", "hardware_score", "ticker"}.issubset(success.columns):
        return []
    cache_dir = base / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return []
    figures = base / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(success["analytic_score"], success["hardware_score"], color="#4aa785", edgecolor="#111111", linewidth=0.6)
    low = float(min(success["analytic_score"].min(), success["hardware_score"].min()))
    high = float(max(success["analytic_score"].max(), success["hardware_score"].max()))
    ax.plot([low, high], [low, high], color="#d77768", linestyle="--", linewidth=1.0, label="analytic = hardware")
    for row in success.itertuples(index=False):
        ax.annotate(str(row.ticker), (row.analytic_score, row.hardware_score), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_title("IBM hardware scores vs analytic simulation")
    ax.set_xlabel("Analytic QNN score")
    ax.set_ylabel("IBM hardware score")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figures / "hardware_score_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path]


def _write_manifest(base: Path, transpile: pd.DataFrame, scores: pd.DataFrame, figure_paths: list[Path]) -> dict:
    existing = _read_manifest(base / "hardware_run_manifest.json")
    success_scores = scores[scores.get("status", "") == "success"].copy() if not scores.empty else pd.DataFrame()
    success_transpile = transpile[transpile.get("status", "") == "success"].copy() if not transpile.empty else pd.DataFrame()
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_type": "ibm_quantum_hardware_robustness_audit",
        "output_dir": base.name,
        "instance_name": os.environ.get("QISKIT_IBM_INSTANCE", ""),
        "claim_guardrail": "Inference-only systems diagnostic; no quantum advantage, trading edge, or real-hardware finance-signal validation claim.",
        "backend_names": _unique_strings(scores, "backend_name") or _unique_strings(transpile, "backend_name"),
        "job_ids": _unique_strings(success_scores, "job_id"),
        "calibration_timestamps": _unique_strings(scores, "backend_calibration_timestamp") or _unique_strings(transpile, "backend_calibration_timestamp"),
        "shots": _unique_values(success_scores, "shots"),
        "requested_precision": _unique_values(success_scores, "requested_precision"),
        "optimization_levels": _unique_values(scores, "optimization_level") or _unique_values(transpile, "optimization_level"),
        "resilience_levels": _unique_values(success_scores, "resilience_level"),
        "python_version": sys.version.split()[0],
        "dependency_versions": _dependency_versions(),
        "artifact_hashes": _artifact_hashes(base, figure_paths),
        "transpilation_summary": _transpilation_manifest(success_transpile),
        "hardware_score_summary": _score_diagnostic_rows(success_scores),
        "top3_comparison": _top3_comparison(success_scores),
        "execution_notes": existing.get("execution_notes", []),
    }
    (base / "hardware_run_manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return manifest


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _unique_strings(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame:
        return []
    return sorted(str(value) for value in frame[column].dropna().unique() if str(value))


def _unique_values(frame: pd.DataFrame, column: str) -> list[object]:
    if frame.empty or column not in frame:
        return []
    values: list[object] = []
    for value in frame[column].dropna().unique():
        if isinstance(value, np.integer):
            values.append(int(value))
        elif isinstance(value, np.floating):
            values.append(float(value))
        else:
            values.append(value.item() if hasattr(value, "item") else value)
    return sorted(values, key=lambda item: str(item))


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ["numpy", "pandas", "qiskit", "qiskit-ibm-runtime", "pennylane", "scipy", "matplotlib"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _artifact_hashes(base: Path, figure_paths: list[Path]) -> dict[str, str]:
    paths = [
        base / "frozen_qnn_subset.json",
        base / "ibm_transpilation_audit.csv",
        base / "ibm_hardware_scores.csv",
        *figure_paths,
    ]
    return {path.relative_to(base).as_posix(): _sha256(path) for path in paths if path.exists()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _transpilation_manifest(success: pd.DataFrame) -> dict[str, object]:
    if success.empty:
        return {}
    return {
        "rows": int(len(success)),
        "pre_depth_mean": float(success["pre_depth"].mean()) if "pre_depth" in success else math.nan,
        "post_depth_mean": float(success["post_depth"].mean()) if "post_depth" in success else math.nan,
        "pre_two_qubit_gate_count_mean": float(success["pre_two_qubit_gate_count"].mean()) if "pre_two_qubit_gate_count" in success else math.nan,
        "post_two_qubit_gate_count_mean": float(success["post_two_qubit_gate_count"].mean()) if "post_two_qubit_gate_count" in success else math.nan,
        "post_swap_count_mean": float(success["post_swap_count"].mean()) if "post_swap_count" in success else math.nan,
    }


def _top3_comparison(success: pd.DataFrame) -> list[dict[str, object]]:
    required = {"model_id", "date", "ticker", "analytic_score", "hardware_score"}
    if success.empty or not required.issubset(success.columns):
        return []
    rows = []
    for (model_id, date), group in success.groupby(["model_id", "date"], dropna=False):
        analytic_top3 = group.sort_values(["analytic_score", "ticker"], ascending=[False, True]).head(3)["ticker"].tolist()
        hardware_top3 = group.sort_values(["hardware_score", "ticker"], ascending=[False, True]).head(3)["ticker"].tolist()
        rows.append(
            {
                "model_id": model_id,
                "date": date,
                "analytic_top3": analytic_top3,
                "hardware_top3": hardware_top3,
                "top3_overlap": len(set(analytic_top3) & set(hardware_top3)) / 3.0,
            }
        )
    return rows


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    columns = [str(column) for column in display.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row.tolist()) + " |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing hardware audit CSVs.")
    args = parser.parse_args(argv)
    make_hardware_report(args.input_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
