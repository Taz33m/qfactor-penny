"""Safe metrics for tiny financial walk-forward samples."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray, *, context: str) -> float:
    if len(y_true) < 2 or len(np.unique(y_true)) < 2 or len(np.unique(scores)) < 2:
        warnings.warn(f"ROC-AUC undefined for {context}; returning NaN.", RuntimeWarning)
        return math.nan
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception as exc:
        warnings.warn(f"ROC-AUC failed for {context}: {exc}; returning NaN.", RuntimeWarning)
        return math.nan


def safe_spearman(x: np.ndarray, y: np.ndarray, *, context: str) -> float:
    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        warnings.warn(f"Spearman rank IC undefined for {context}; returning NaN.", RuntimeWarning)
        return math.nan
    ranked_x = pd.Series(x).rank(method="average").to_numpy()
    ranked_y = pd.Series(y).rank(method="average").to_numpy()
    corr = np.corrcoef(ranked_x, ranked_y)[0, 1]
    if not np.isfinite(corr):
        warnings.warn(f"Spearman rank IC non-finite for {context}; returning NaN.", RuntimeWarning)
        return math.nan
    return float(corr)


def safe_sharpe(returns: np.ndarray, *, context: str) -> float:
    clean = np.asarray([value for value in returns if np.isfinite(value)], dtype=float)
    if len(clean) < 2:
        warnings.warn(f"Sharpe undefined for {context}; returning NaN.", RuntimeWarning)
        return math.nan
    vol = float(np.std(clean, ddof=1))
    if vol == 0.0 or not np.isfinite(vol):
        warnings.warn(f"Sharpe zero-volatility for {context}; returning NaN.", RuntimeWarning)
        return math.nan
    return float(np.mean(clean) / vol * np.sqrt(252 / 5))


def safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, *, context: str) -> float:
    if len(y_true) < 2 or len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        warnings.warn(f"Balanced accuracy undefined for {context}; returning NaN.", RuntimeWarning)
        return math.nan
    try:
        return float(balanced_accuracy_score(y_true, y_pred))
    except Exception as exc:
        warnings.warn(f"Balanced accuracy failed for {context}: {exc}; returning NaN.", RuntimeWarning)
        return math.nan


def safe_f1(y_true: np.ndarray, y_pred: np.ndarray, *, context: str) -> float:
    if len(y_true) < 2 or len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        warnings.warn(f"F1 undefined for {context}; returning NaN.", RuntimeWarning)
        return math.nan
    try:
        return float(f1_score(y_true, y_pred, zero_division=0))
    except Exception as exc:
        warnings.warn(f"F1 failed for {context}: {exc}; returning NaN.", RuntimeWarning)
        return math.nan


def safe_max_drawdown(returns: np.ndarray, *, context: str) -> float:
    clean = np.asarray([value for value in returns if np.isfinite(value)], dtype=float)
    if len(clean) == 0:
        warnings.warn(f"Max drawdown undefined for {context}; returning NaN.", RuntimeWarning)
        return math.nan
    equity = np.cumprod(1.0 + clean)
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    return float(np.min(drawdowns))


def evaluate_model_predictions(predictions: pd.DataFrame, *, model: str, split_id: str) -> dict[str, object]:
    labeled = predictions[predictions["label"].notna()].copy()
    y_true = labeled["label"].to_numpy(dtype=int)
    scores = labeled["score"].to_numpy(dtype=float)
    roc_auc = safe_roc_auc(y_true, scores, context=f"{model}/{split_id}")
    threshold = float(np.nanmedian(scores)) if len(scores) else math.nan
    if len(y_true) and np.isfinite(threshold):
        y_pred = (scores >= threshold).astype(int)
        balanced_accuracy = safe_balanced_accuracy(y_true, y_pred, context=f"{model}/{split_id}")
        f1 = safe_f1(y_true, y_pred, context=f"{model}/{split_id}")
    else:
        warnings.warn(f"Classification threshold undefined for {model}/{split_id}; returning NaN.", RuntimeWarning)
        balanced_accuracy = math.nan
        f1 = math.nan

    precision_values = []
    rank_ic_values = []
    for date, group in predictions.groupby("date"):
        ranked = group.sort_values(["score", "ticker"], ascending=[False, True])
        top = ranked.head(3)
        precision_values.append(float((top["label"] == 1.0).sum() / 3.0))
        rank_ic_values.append(
            safe_spearman(
                group["score"].to_numpy(dtype=float),
                group["excess_return_5d"].to_numpy(dtype=float),
                context=f"{model}/{split_id}/{date}",
            )
        )
    return {
        "split_id": split_id,
        "seed": int(predictions["seed"].iloc[0]) if "seed" in predictions.columns and len(predictions) else np.nan,
        "model": model,
        "rows": int(len(predictions)),
        "labeled_rows": int(len(labeled)),
        "roc_auc": roc_auc,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "precision_at_3": float(np.nanmean(precision_values)) if precision_values else math.nan,
        "rank_ic": float(np.nanmean(rank_ic_values)) if rank_ic_values else math.nan,
    }
