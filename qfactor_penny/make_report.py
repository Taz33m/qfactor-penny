"""Create QFactor-Penny markdown report and figures."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config, project_path

FRAMING = (
    "This project evaluates whether small trainable PennyLane QNNs provide stable or useful ranking behavior "
    "under walk-forward financial validation. It compares QNNs against strong classical baselines and reports "
    "limitations, failure cases, and sensitivity to circuit design and shots. It does not claim quantum advantage."
)


def _format_float(value: object) -> str:
    try:
        number = float(value)
    except Exception:
        return ""
    if not math.isfinite(number):
        return "NaN"
    return f"{number:.4f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(_format_float)
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


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


def _model_aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
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


def _portfolio_aggregate(portfolio: pd.DataFrame) -> pd.DataFrame:
    return _mean_std(
        portfolio,
        group_by=["model"],
        columns=["gross_return", "net_return", "alpha_vs_spy", "turnover", "transaction_cost"],
    )


def _undefined_metric_audit(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.isna()
        .sum()
        .rename_axis("metric")
        .reset_index(name="nan_count")
        .query("nan_count > 0")
    )


def _seed_stability_summary(metrics: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in metrics.groupby("model"):
        seeds = sorted(group["seed"].dropna().unique()) if "seed" in group else []
        row: dict[str, object] = {
            "model": model,
            "num_seeds": len(seeds),
            "rank_ic_mean": float(group["rank_ic"].mean()),
            "rank_ic_std": float(group["rank_ic"].std()),
            "rank_ic_nan_count": int(group["rank_ic"].isna().sum()),
        }
        if len(seeds) >= 2:
            pivot = group.pivot_table(index="split_id", columns="seed", values="rank_ic", aggfunc="mean")
            if len(pivot.columns) >= 2:
                first, second = pivot.columns[:2]
                deltas = (pivot[first] - pivot[second]).abs()
                row["rank_ic_mean_abs_seed_delta"] = float(deltas.mean())
                row["rank_ic_max_abs_seed_delta"] = float(deltas.max())
        model_portfolio = portfolio[portfolio["model"] == model]
        if not model_portfolio.empty:
            by_seed = model_portfolio.groupby("seed")["net_return"].mean()
            row["portfolio_net_return_seed_min"] = float(by_seed.min())
            row["portfolio_net_return_seed_max"] = float(by_seed.max())
            row["portfolio_net_return_seed_range"] = float(by_seed.max() - by_seed.min())
            selection_pivot = model_portfolio.pivot_table(
                index=["split_id", "date"],
                columns="seed",
                values="selected_tickers",
                aggfunc="first",
            )
            overlaps: list[int] = []
            same: list[bool] = []
            for selection in selection_pivot.dropna().itertuples(index=False):
                if len(selection) < 2:
                    continue
                left = set(str(selection[0]).split(","))
                right = set(str(selection[1]).split(","))
                overlaps.append(len(left & right))
                same.append(left == right)
            if overlaps:
                row["mean_top3_seed_overlap"] = float(np.mean(overlaps))
                row["same_top3_seed_rate"] = float(np.mean(same))
        rows.append(row)
    return pd.DataFrame(rows)


def _path_dependence_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in portfolio.groupby("model"):
        total = float(group["net_return"].sum())
        split_returns = group.groupby("split_id")["net_return"].sum()
        date_returns = group.groupby("date")["net_return"].sum()
        best_split = split_returns.idxmax()
        worst_split = split_returns.idxmin()
        best_date = date_returns.idxmax()
        worst_date = date_returns.idxmin()
        rows.append(
            {
                "model": model,
                "mean_net_return": float(group["net_return"].mean()),
                "mean_alpha_vs_spy": float(group["alpha_vs_spy"].mean()),
                "mean_turnover": float(group["turnover"].mean()),
                "total_net_return": total,
                "best_split": best_split,
                "best_split_net_return": float(split_returns.max()),
                "best_split_share_of_total": float(split_returns.max() / total) if total else math.nan,
                "mean_net_return_ex_best_split": float(group[group["split_id"] != best_split]["net_return"].mean()),
                "worst_split": worst_split,
                "worst_split_net_return": float(split_returns.min()),
                "best_date": best_date,
                "best_date_net_return": float(date_returns.max()),
                "best_date_share_of_total": float(date_returns.max() / total) if total else math.nan,
                "mean_net_return_ex_best_date": float(group[group["date"] != best_date]["net_return"].mean()),
                "worst_date": worst_date,
                "worst_date_net_return": float(date_returns.min()),
                "positive_alpha_rate": float((group["alpha_vs_spy"] > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _qnn_failure_audit(predictions: pd.DataFrame, metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    qnn = predictions[predictions["model"] == "pennylane_qnn"].copy()
    if qnn.empty:
        return pd.DataFrame()
    score_groups = (
        qnn.groupby(["split_id", "date", "seed"])["score"]
        .agg(num_scores="count", unique_scores="nunique", score_std="std", min_score="min", max_score="max")
        .reset_index()
    )
    score_groups["constant_score_group"] = score_groups["unique_scores"].le(1)
    split_summary = (
        score_groups.groupby(["split_id", "seed"], as_index=False)
        .agg(
            qnn_date_groups=("date", "count"),
            constant_score_groups=("constant_score_group", "sum"),
            mean_score_std=("score_std", "mean"),
            min_unique_scores=("unique_scores", "min"),
            max_unique_scores=("unique_scores", "max"),
        )
    )
    metric_columns = ["split_id", "seed", "roc_auc", "balanced_accuracy", "f1", "precision_at_3", "rank_ic"]
    split_summary = split_summary.merge(
        metrics[metrics["model"] == "pennylane_qnn"][metric_columns],
        on=["split_id", "seed"],
        how="left",
    )
    if not diagnostics.empty:
        diagnostic_columns = [
            "split_id",
            "seed",
            "train_seconds",
            "inference_seconds",
            "shot_score_correlation",
            "shot_mean_abs_score_diff",
            "shot_ranking_flip_rate",
            "epochs_ran",
            "best_validation_loss",
            "selected_features",
        ]
        split_summary = split_summary.merge(
            diagnostics[diagnostic_columns],
            on=["split_id", "seed"],
            how="left",
        )
    split_summary["any_undefined_metric"] = split_summary[["roc_auc", "balanced_accuracy", "f1", "rank_ic"]].isna().any(axis=1)
    return split_summary


def _portfolio_selection_audit(predictions: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prediction_groups = predictions.groupby(["split_id", "date", "model", "seed"])
    for row in portfolio.itertuples(index=False):
        if row.model in {"spy_benchmark", "equal_weight_sector"}:
            continue
        key = (row.split_id, row.date, row.model, row.seed)
        if key not in prediction_groups.groups:
            rows.append(
                {
                    "split_id": row.split_id,
                    "date": row.date,
                    "model": row.model,
                    "seed": row.seed,
                    "selected_tickers": row.selected_tickers,
                    "expected_top3_tickers": "",
                    "selection_matches_score_rank": False,
                }
            )
            continue
        group = prediction_groups.get_group(key)
        expected = ",".join(group.sort_values(["score", "ticker"], ascending=[False, True]).head(3)["ticker"])
        rows.append(
            {
                "split_id": row.split_id,
                "date": row.date,
                "model": row.model,
                "seed": row.seed,
                "selected_tickers": row.selected_tickers,
                "expected_top3_tickers": expected,
                "selection_matches_score_rank": row.selected_tickers == expected,
            }
        )
    return pd.DataFrame(rows)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_manifest(results_dir: Path) -> dict[str, Any]:
    path = results_dir / "run_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _has_current_variant_schema(results_dir: Path, manifest: dict[str, Any]) -> bool:
    feature_selection = manifest.get("feature_selection", {})
    mode = feature_selection.get("mode") if isinstance(feature_selection, dict) else None
    if mode not in {"standard", "cross_sectional_aware"}:
        return False
    required_files = [
        results_dir / "run_manifest.json",
        results_dir / "model_run_status.csv",
        results_dir / "feature_stability_summary.csv",
        results_dir / "qnn_failure_audit.csv",
    ]
    if not all(path.exists() for path in required_files):
        return False
    status = _read_optional_csv(results_dir / "model_run_status.csv")
    feature_stability = _read_optional_csv(results_dir / "feature_stability_summary.csv")
    required_status = {"split_id", "seed", "model", "status", "num_constant_score_dates", "diagnostics_available"}
    required_feature_stability = {"split_id", "seed", "selected_feature", "feature_selection_mode", "is_calendar_heavy"}
    return (
        not status.empty
        and not feature_stability.empty
        and required_status.issubset(status.columns)
        and required_feature_stability.issubset(feature_stability.columns)
    )


def _model_run_status_summary(status: pd.DataFrame) -> pd.DataFrame:
    if status.empty:
        return pd.DataFrame()
    return (
        status.groupby(["model", "status"], as_index=False)
        .agg(
            rows=("status", "size"),
            total_constant_score_dates=("num_constant_score_dates", "sum"),
            total_nan_scores=("num_nan_scores", "sum"),
            diagnostics_available_rows=("diagnostics_available", "sum"),
        )
        .sort_values(["model", "status"])
    )


def _feature_stability_summary(feature_stability: pd.DataFrame) -> pd.DataFrame:
    if feature_stability.empty:
        return pd.DataFrame()
    return (
        feature_stability.groupby(["selected_feature", "selected_for_model"], as_index=False)
        .agg(
            selected_count=("selected_feature", "size"),
            calendar_heavy_rate=("is_calendar_heavy", "mean"),
            mean_cross_sectional_std_by_date=("mean_cross_sectional_std_by_date", "mean"),
            mean_time_series_std_by_ticker=("mean_time_series_std_by_ticker", "mean"),
            mean_variance_ratio=("cross_sectional_to_time_series_variance_ratio", "mean"),
            mean_missing_rate=("missing_rate", "mean"),
        )
        .sort_values(["calendar_heavy_rate", "selected_count"], ascending=[False, False])
    )


def _variant_comparison(base: Path, current_results_dir: Path) -> pd.DataFrame:
    candidate_names = [
        "results",
        "results_cross_sectional_mvp",
        "results_full",
        "results_cross_sectional",
        "results_cross_sectional_features",
        current_results_dir.name,
    ]
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for name in candidate_names:
        results_dir = (base / name).resolve()
        if results_dir in seen or not results_dir.exists():
            continue
        seen.add(results_dir)
        metrics_path = results_dir / "metrics_summary.csv"
        portfolio_path = results_dir / "portfolio_summary.csv"
        manifest = _read_manifest(results_dir)
        if not metrics_path.exists() or not portfolio_path.exists():
            continue
        if not _has_current_variant_schema(results_dir, manifest):
            continue
        metrics = pd.read_csv(metrics_path)
        portfolio = pd.read_csv(portfolio_path)
        qnn_metrics = metrics[metrics["model"] == "pennylane_qnn"]
        qnn_failure = _read_optional_csv(results_dir / "qnn_failure_audit.csv")
        feature_stability = _read_optional_csv(results_dir / "feature_stability_summary.csv")
        feature_selection = manifest.get("feature_selection", {})
        mode = feature_selection.get("mode", "unknown") if isinstance(feature_selection, dict) else "unknown"
        qnn_undefined = (
            int(qnn_metrics[["roc_auc", "balanced_accuracy", "f1", "rank_ic"]].isna().any(axis=1).sum())
            if not qnn_metrics.empty
            else 0
        )
        qnn_constant = (
            int(qnn_failure["constant_score_groups"].sum())
            if not qnn_failure.empty and "constant_score_groups" in qnn_failure
            else 0
        )
        qnn_failure_rows = int(qnn_failure["any_undefined_metric"].sum()) if not qnn_failure.empty and "any_undefined_metric" in qnn_failure else 0
        calendar_count = int(feature_stability["is_calendar_heavy"].sum()) if not feature_stability.empty else 0
        feature_count = int(len(feature_stability)) if not feature_stability.empty else 0
        calendar_share = float(calendar_count / feature_count) if feature_count else math.nan
        metrics_by_model = metrics.groupby("model", as_index=False).agg(
            rank_ic_mean=("rank_ic", "mean"),
            precision_at_3_mean=("precision_at_3", "mean"),
        )
        portfolio_by_model = portfolio.groupby("model", as_index=False).agg(
            portfolio_net_return_mean=("net_return", "mean"),
            alpha_vs_spy_mean=("alpha_vs_spy", "mean"),
            turnover_mean=("turnover", "mean"),
        )
        comparison = metrics_by_model.merge(portfolio_by_model, on="model", how="left")
        for row in comparison.itertuples(index=False):
            rows.append(
                {
                    "results_dir": results_dir.name,
                    "feature_selection_mode": mode,
                    "max_splits": manifest.get("max_splits"),
                    "seeds": ",".join(str(seed) for seed in manifest.get("random_seeds", [])),
                    "model": row.model,
                    "rank_ic_mean": row.rank_ic_mean,
                    "precision_at_3_mean": row.precision_at_3_mean,
                    "portfolio_net_return_mean": row.portfolio_net_return_mean,
                    "alpha_vs_spy_mean": row.alpha_vs_spy_mean,
                    "turnover_mean": row.turnover_mean,
                    "qnn_constant_groups": qnn_constant,
                    "qnn_failure_rows": qnn_failure_rows,
                    "qnn_undefined_metric_rows": qnn_undefined,
                    "calendar_heavy_selected_features": calendar_count,
                    "selected_feature_rows": feature_count,
                    "calendar_heavy_selected_feature_share": calendar_share,
                }
            )
    return pd.DataFrame(rows)


def _repo_relative(path: Path, base: Path, fallback_base: Path | None = None) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        if fallback_base is not None:
            try:
                return path.resolve().relative_to(fallback_base.resolve()).as_posix()
            except ValueError:
                pass
        return path.name


def _variant_interpretation(variant_comparison: pd.DataFrame) -> str:
    if variant_comparison.empty or variant_comparison["feature_selection_mode"].nunique() < 2:
        return "No paired standard and cross-sectional-aware runs were available for interpretation."
    qnn = variant_comparison[variant_comparison["model"] == "pennylane_qnn"].copy()
    standard = qnn[qnn["feature_selection_mode"] == "standard"]
    cross = qnn[qnn["feature_selection_mode"] == "cross_sectional_aware"]
    if standard.empty or cross.empty:
        return "Variant outputs are present, but a paired QNN standard/cross-sectional-aware comparison was not available."
    standard_row = standard.iloc[0]
    cross_row = cross.iloc[0]
    collapse_delta = cross_row["qnn_constant_groups"] - standard_row["qnn_constant_groups"]
    rank_delta = cross_row["rank_ic_mean"] - standard_row["rank_ic_mean"]
    precision_delta = cross_row["precision_at_3_mean"] - standard_row["precision_at_3_mean"]
    alpha_delta = cross_row["alpha_vs_spy_mean"] - standard_row["alpha_vs_spy_mean"]
    collapse_text = "reduced" if collapse_delta < 0 else "increased" if collapse_delta > 0 else "did not change"
    rank_text = "improved" if rank_delta > 0 else "worsened" if rank_delta < 0 else "did not change"
    precision_text = "improved" if precision_delta > 0 else "worsened" if precision_delta < 0 else "did not change"
    alpha_text = "improved" if alpha_delta > 0 else "worsened" if alpha_delta < 0 else "did not change"
    return (
        "Cross-sectional-aware selection is treated as a diagnostic variant, not post-hoc performance tuning. "
        f"For the QNN, constant-score groups {collapse_text} ({standard_row['qnn_constant_groups']} -> {cross_row['qnn_constant_groups']}), "
        f"rank IC {rank_text} ({_format_float(standard_row['rank_ic_mean'])} -> {_format_float(cross_row['rank_ic_mean'])}), "
        f"precision@3 {precision_text} ({_format_float(standard_row['precision_at_3_mean'])} -> {_format_float(cross_row['precision_at_3_mean'])}), "
        f"and alpha vs SPY after costs {alpha_text} ({_format_float(standard_row['alpha_vs_spy_mean'])} -> {_format_float(cross_row['alpha_vs_spy_mean'])}). "
        "A positive diagnostic movement should still not be read as quantum advantage or a trading result."
    )


def _write_figures(results_dir: Path, metrics: pd.DataFrame, portfolio: pd.DataFrame, diagnostics: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    cache_dir = results_dir / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(results_dir / ".cache"))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return paths
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    if not metrics.empty:
        summary = metrics.groupby("model", as_index=False).agg(rank_ic=("rank_ic", "mean"), roc_auc=("roc_auc", "mean"))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(summary["model"], summary["rank_ic"])
        ax.set_title("Mean rank IC by model")
        ax.set_ylabel("Rank IC")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = figures / "model_rank_ic.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(summary["model"], summary["roc_auc"])
        ax.axhline(0.5, color="#666666", linewidth=1, linestyle="--")
        ax.set_title("Mean ROC-AUC by model")
        ax.set_ylabel("ROC-AUC")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = figures / "roc_auc_by_model.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

        split_model = metrics.groupby(["split_id", "model"], as_index=False).agg(rank_ic=("rank_ic", "mean"))
        fig, ax = plt.subplots(figsize=(11, 5))
        for model, group in split_model.groupby("model"):
            ordered = group.sort_values("split_id")
            ax.plot(ordered["split_id"], ordered["rank_ic"], marker="o", label=model)
        ax.axhline(0.0, color="#666666", linewidth=1)
        ax.set_title("Rank IC by split and model")
        ax.set_ylabel("Rank IC")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = figures / "split_rank_ic_by_model.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    if not portfolio.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for model, group in portfolio.groupby("model"):
            ordered = group.groupby("date", as_index=False).agg(net_return=("net_return", "mean")).sort_values("date")
            equity = (1.0 + ordered["net_return"].fillna(0.0)).cumprod()
            ax.plot(pd.to_datetime(ordered["date"]), equity, label=model)
        ax.set_title("Portfolio equity curves")
        ax.set_ylabel("Growth of $1")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = figures / "portfolio_equity.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

        portfolio_summary = portfolio.groupby("model", as_index=False).agg(
            alpha_vs_spy=("alpha_vs_spy", "mean"),
            net_return=("net_return", "mean"),
            turnover=("turnover", "mean"),
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(portfolio_summary["model"], portfolio_summary["alpha_vs_spy"])
        ax.axhline(0.0, color="#666666", linewidth=1)
        ax.set_title("Mean alpha vs SPY by model")
        ax.set_ylabel("Net return minus SPY return")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = figures / "alpha_vs_spy_by_model.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(portfolio_summary["turnover"], portfolio_summary["net_return"])
        for row in portfolio_summary.itertuples(index=False):
            ax.annotate(row.model, (row.turnover, row.net_return), fontsize=7)
        ax.set_title("Turnover vs mean net return")
        ax.set_xlabel("Mean turnover")
        ax.set_ylabel("Mean net return")
        fig.tight_layout()
        path = figures / "turnover_vs_return.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    if not diagnostics.empty and "shot_mean_abs_score_diff" in diagnostics:
        fig, ax = plt.subplots(figsize=(9, 5))
        diag = diagnostics.sort_values(["seed", "split_id"])
        x_labels = [f"{row.split_id}\nseed {row.seed}" if "seed" in diagnostics.columns else row.split_id for row in diag.itertuples()]
        ax.bar(x_labels, diag["shot_mean_abs_score_diff"])
        ax.set_title("QNN 1024-shot sensitivity")
        ax.set_ylabel("Mean absolute score difference")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = figures / "qnn_shot_sensitivity.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def make_report(config: dict[str, Any], *, root: str | Path | None = None) -> Path:
    base = Path.cwd() if root is None else Path(root)
    results_dir = project_path(config.get("results_dir", "results"), root=base)
    metrics = pd.read_csv(results_dir / "metrics_summary.csv")
    portfolio = pd.read_csv(results_dir / "portfolio_summary.csv")
    predictions = pd.read_csv(results_dir / "rebalance_predictions.csv")
    diagnostics = pd.read_csv(results_dir / "quantum_diagnostics.csv")
    split_audit = pd.read_csv(results_dir / "split_audit.csv")
    model_status = _read_optional_csv(results_dir / "model_run_status.csv")
    feature_stability = _read_optional_csv(results_dir / "feature_stability_summary.csv")
    manifest = _read_manifest(results_dir)
    prediction_audit_path = results_dir / "prediction_audit.csv"
    prediction_audit = pd.read_csv(prediction_audit_path) if prediction_audit_path.exists() else pd.DataFrame()
    model_aggregate = _model_aggregate(metrics)
    portfolio_aggregate = _portfolio_aggregate(portfolio)
    model_status_summary = _model_run_status_summary(model_status)
    feature_stability_aggregate = _feature_stability_summary(feature_stability)
    nan_audit = _undefined_metric_audit(metrics)
    seed_stability = _seed_stability_summary(metrics, portfolio)
    path_dependence = _path_dependence_summary(portfolio)
    qnn_failure_audit = _qnn_failure_audit(predictions, metrics, diagnostics)
    portfolio_selection_audit = _portfolio_selection_audit(predictions, portfolio)
    model_aggregate.to_csv(results_dir / "model_aggregate_summary.csv", index=False)
    portfolio_aggregate.to_csv(results_dir / "portfolio_aggregate_summary.csv", index=False)
    if not model_status_summary.empty:
        model_status_summary.to_csv(results_dir / "model_run_status_summary.csv", index=False)
    if not feature_stability_aggregate.empty:
        feature_stability_aggregate.to_csv(results_dir / "feature_stability_aggregate.csv", index=False)
    nan_audit.to_csv(results_dir / "undefined_metric_audit.csv", index=False)
    seed_stability.to_csv(results_dir / "seed_stability_summary.csv", index=False)
    path_dependence.to_csv(results_dir / "path_dependence_summary.csv", index=False)
    qnn_failure_audit.to_csv(results_dir / "qnn_failure_audit.csv", index=False)
    portfolio_selection_audit.to_csv(results_dir / "portfolio_selection_audit.csv", index=False)
    variant_comparison = _variant_comparison(base, results_dir)
    variant_comparison.to_csv(results_dir / "experimental_variant_comparison.csv", index=False)
    figure_paths = _write_figures(results_dir, metrics, portfolio, diagnostics)

    lines = [
        "# QFactor-Penny Results Summary",
        "",
        FRAMING,
        "",
        "## Executive Summary",
        "",
        f"This run evaluated `{metrics['model'].nunique()}` model families across `{split_audit['split_id'].nunique()}` walk-forward splits and `{metrics['seed'].nunique() if 'seed' in metrics else 1}` seed(s). The report is descriptive: it compares ranking stability, portfolio accounting, and QNN shot sensitivity without asserting a tradable edge or quantum advantage.",
        "",
        "## Methodology",
        "",
        "The benchmark uses non-overlapping five-trading-day rebalance dates. Labels are SPY-relative: top 3 sectors are labeled 1, bottom 3 sectors are labeled 0, and middle sectors are dropped from classification training. Portfolio returns use absolute realized ETF returns.",
        "",
        f"This report was generated from results directory `{results_dir.name}` with config split cap `{config.get('max_splits', 'none')}` and seeds `{config.get('seeds', [config.get('random_state', 42)])}`. `configs/mvp.yaml` is the fast smoke path; `configs/full.yaml` is the slower research path.",
        "",
        "## Dataset And Coverage",
        "",
        f"- Metric rows: `{len(metrics)}`",
        f"- Portfolio rows: `{len(portfolio)}`",
        f"- QNN diagnostic rows: `{len(diagnostics)}`",
        f"- Split date range: `{split_audit['train_start'].min()}` to `{split_audit['test_end'].max()}`",
        f"- Config hash: `{manifest.get('config_hash', 'unavailable')}`",
        f"- Dataset hash: `{manifest.get('dataset_hash', 'unavailable')}`",
        f"- Runtime seconds: `{_format_float(manifest.get('runtime_seconds', math.nan))}`",
        "",
        "## Leakage Audit Interpretation",
        "",
        "The split audit records train, validation, and test date boundaries plus the final realized target-window end before validation and test. Valid rows should have `train_last_forward_end < validation_start` and `validation_last_forward_end < test_start`; this report preserves those columns for manual review.",
        "",
        "## Split Audit",
        "",
        _markdown_table(split_audit),
        "",
        "## Prediction Shape Audit",
        "",
        "Every model/date/seed group should score all 11 sector ETFs while retaining exactly 6 labeled training-comparable rows and 5 middle-sector inference rows. `model_rank_position` should match the model's own score ordering; realized future rank is reported separately as `realized_rank_position` in the prediction CSV.",
        "",
        _markdown_table(prediction_audit.head(20)) if not prediction_audit.empty else "_No prediction audit rows._",
        "",
        "## Model Run Status",
        "",
        "Failed, skipped, constant-score, or diagnostics-missing runs should be visible rather than silently absorbed into aggregates.",
        "",
        _markdown_table(model_status_summary) if not model_status_summary.empty else "_No model-run status rows._",
        "",
        "## Portfolio Selection Audit",
        "",
        "Portfolio top-3 selections should match the score-ranked prediction table with deterministic ticker tie-breaking.",
        "",
        _markdown_table(portfolio_selection_audit.groupby("selection_matches_score_rank", as_index=False).size())
        if not portfolio_selection_audit.empty
        else "_No portfolio selection audit rows._",
        "",
        "## Undefined Metric Audit",
        "",
        _markdown_table(nan_audit) if not nan_audit.empty else "No undefined model-level metrics were produced in this run.",
        "",
        "## Seed Stability Audit",
        "",
        _markdown_table(seed_stability.sort_values("rank_ic_mean", ascending=False) if "rank_ic_mean" in seed_stability else seed_stability),
        "",
        "## Feature Stability Audit",
        "",
        "Calendar-heavy selected features have low within-date cross-sectional dispersion relative to their time-series dispersion; these can produce weak cross-sectional ranking behavior.",
        "",
        _markdown_table(feature_stability_aggregate.head(20)) if not feature_stability_aggregate.empty else "_No feature-stability rows._",
        "",
        "## Path Dependence Audit",
        "",
        _markdown_table(path_dependence.sort_values("mean_net_return", ascending=False) if "mean_net_return" in path_dependence else path_dependence),
        "",
        "## Model Aggregate Summary",
        "",
        _markdown_table(model_aggregate.sort_values("rank_ic_mean", ascending=False) if "rank_ic_mean" in model_aggregate else model_aggregate),
        "",
        "## Model Metrics",
        "",
        _markdown_table(metrics.sort_values(["rank_ic", "roc_auc"], ascending=False).head(20)),
        "",
        "## Portfolio Aggregate Summary",
        "",
        _markdown_table(portfolio_aggregate.sort_values("alpha_vs_spy_mean", ascending=False) if "alpha_vs_spy_mean" in portfolio_aggregate else portfolio_aggregate),
        "",
        "## Portfolio Summary",
        "",
        _markdown_table(
            portfolio.groupby("model", as_index=False).agg(
                net_return_mean=("net_return", "mean"),
                alpha_vs_spy_mean=("alpha_vs_spy", "mean"),
                turnover_mean=("turnover", "mean"),
            )
        ),
        "",
        "## Quantum Diagnostics",
        "",
        _markdown_table(diagnostics) if not diagnostics.empty else "No PennyLane QNN diagnostics were produced.",
        "",
        "## QNN Shot-Sensitivity Interpretation",
        "",
        "The QNN is trained analytically, then a fixed subset of inference samples is re-evaluated with 1024-shot sampling. Large score differences, weak score correlation, or high pairwise ranking flip rates indicate sensitivity to finite-shot execution and should be treated as a limitation rather than a positive result.",
        "",
        "## QNN Failure Audit",
        "",
        _markdown_table(qnn_failure_audit) if not qnn_failure_audit.empty else "No QNN failure-audit rows were produced.",
        "",
        "## Experimental Variant Comparison",
        "",
        "If both standard and cross-sectional-aware feature-selection runs are available, this table compares whether the variant reduced QNN constant-score collapse, improved rank IC or precision@3, helped models beyond QNN, and survived transaction costs.",
        "",
        _variant_interpretation(variant_comparison),
        "",
        _markdown_table(variant_comparison) if len(variant_comparison) > 1 else "No alternate feature-selection run was available for comparison.",
        "",
        "## Figures",
        "",
    ]
    if figure_paths:
        lines.extend([f"- `{_repo_relative(path, base, results_dir.parent)}`" for path in figure_paths])
    else:
        lines.append("- No figures generated.")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "This benchmark is a sensitivity study, not a trading product. Undefined metrics are reported as NaN when samples are too small, scores are constant, labels are one-class, or volatility is zero. Classical baselines may dominate, multi-seed variation may be large, and any QNN behavior should be interpreted as experimental.",
            "",
            "Failure cases matter here: negative rank IC, high turnover, weak shot-score correlation, or classical dominance are all valid benchmark outcomes and should not be hidden.",
            "",
        ]
    )
    output = results_dir / "results_summary.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QFactor-Penny report.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print({"summary": str(make_report(load_config(args.config)))})


if __name__ == "__main__":
    main()
