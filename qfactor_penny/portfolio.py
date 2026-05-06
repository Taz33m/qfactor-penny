"""Portfolio evaluation from rebalance predictions."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .constants import SECTOR_TICKERS
from .metrics import safe_max_drawdown, safe_sharpe


def _turnover(new_weights: dict[str, float], old_weights: dict[str, float]) -> float:
    tickers = set(new_weights) | set(old_weights)
    return float(0.5 * sum(abs(new_weights.get(ticker, 0.0) - old_weights.get(ticker, 0.0)) for ticker in tickers))


def _portfolio_rows_for_model(model_frame: pd.DataFrame, *, transaction_cost: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous: dict[str, float] = {}
    for date, group in model_frame.sort_values("date").groupby("date", sort=True):
        tradable = group[group["ticker"].isin(SECTOR_TICKERS)].copy()
        ranked = tradable.sort_values(["score", "ticker"], ascending=[False, True])
        selected = ranked.head(3)
        weights = {ticker: 1.0 / len(selected) for ticker in selected["ticker"]} if len(selected) else {}
        turnover = _turnover(weights, previous)
        gross = float(np.average(selected["forward_return_5d"])) if len(selected) else math.nan
        cost = turnover * transaction_cost
        net = gross - cost if np.isfinite(gross) else math.nan
        spy_return = float(tradable["spy_forward_return_5d"].mean()) if len(tradable) else math.nan
        equal_weight = float(tradable["forward_return_5d"].mean()) if len(tradable) else math.nan
        rows.append(
            {
                "split_id": str(tradable["split_id"].iloc[0]) if len(tradable) else "",
                "seed": int(tradable["seed"].iloc[0]) if len(tradable) and "seed" in tradable else 0,
                "date": str(pd.Timestamp(date).date()),
                "model": str(tradable["model"].iloc[0]) if len(tradable) else "",
                "selected_tickers": ",".join(weights),
                "gross_return": gross,
                "turnover": turnover,
                "transaction_cost": cost,
                "net_return": net,
                "spy_return": spy_return,
                "equal_weight_sector_return": equal_weight,
                "alpha_vs_spy": net - spy_return if np.isfinite(net) and np.isfinite(spy_return) else math.nan,
            }
        )
        previous = weights
    return rows


def _benchmark_rows(predictions: pd.DataFrame, *, transaction_cost: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seed_column = "seed" if "seed" in predictions.columns else None
    base = predictions.drop_duplicates(["split_id", "date", "ticker", *(["seed"] if seed_column else [])])
    for seed, seed_frame in base.groupby(seed_column) if seed_column else [(0, base)]:
        equal_previous: dict[str, float] = {}
        for date, group in seed_frame.sort_values("date").groupby("date", sort=True):
            tradable = group[group["ticker"].isin(SECTOR_TICKERS)].copy()
            split_id = str(tradable["split_id"].iloc[0]) if len(tradable) else ""
            spy_return = float(tradable["spy_forward_return_5d"].mean()) if len(tradable) else math.nan
            rows.append(
                {
                    "split_id": split_id,
                    "seed": int(seed),
                    "date": str(pd.Timestamp(date).date()),
                    "model": "spy_benchmark",
                    "selected_tickers": "SPY",
                    "gross_return": spy_return,
                    "turnover": 0.0,
                    "transaction_cost": 0.0,
                    "net_return": spy_return,
                    "spy_return": spy_return,
                    "equal_weight_sector_return": float(tradable["forward_return_5d"].mean()) if len(tradable) else math.nan,
                    "alpha_vs_spy": 0.0,
                }
            )
            equal_weights = {ticker: 1.0 / len(tradable) for ticker in tradable["ticker"]} if len(tradable) else {}
            equal_turnover = _turnover(equal_weights, equal_previous)
            equal_gross = float(tradable["forward_return_5d"].mean()) if len(tradable) else math.nan
            equal_cost = equal_turnover * transaction_cost
            equal_net = equal_gross - equal_cost if np.isfinite(equal_gross) else math.nan
            rows.append(
                {
                    "split_id": split_id,
                    "seed": int(seed),
                    "date": str(pd.Timestamp(date).date()),
                    "model": "equal_weight_sector",
                    "selected_tickers": ",".join(equal_weights),
                    "gross_return": equal_gross,
                    "turnover": equal_turnover,
                    "transaction_cost": equal_cost,
                    "net_return": equal_net,
                    "spy_return": spy_return,
                    "equal_weight_sector_return": equal_gross,
                    "alpha_vs_spy": equal_net - spy_return if np.isfinite(equal_net) and np.isfinite(spy_return) else math.nan,
                }
            )
            equal_previous = equal_weights
    return rows


def build_portfolio_summary(predictions: pd.DataFrame, *, transaction_cost: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["model", "seed"] if "seed" in predictions.columns else ["model"]
    for _, model_frame in predictions.groupby(group_columns):
        rows.extend(_portfolio_rows_for_model(model_frame, transaction_cost=transaction_cost))
    rows.extend(_benchmark_rows(predictions, transaction_cost=transaction_cost))
    return pd.DataFrame(rows)


def aggregate_portfolio_metrics(portfolio: pd.DataFrame) -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, float]] = defaultdict(dict)
    group_columns = ["model", "seed"] if "seed" in portfolio.columns else ["model"]
    for key, group in portfolio.groupby(group_columns):
        if isinstance(key, tuple):
            model, seed = key
        else:
            model, seed = key, 0
        returns = group["net_return"].to_numpy(dtype=float)
        aggregates[(model, int(seed))] = {
            "portfolio_net_return_mean": float(np.nanmean(returns)) if len(returns) else math.nan,
            "portfolio_alpha_mean": float(np.nanmean(group["alpha_vs_spy"].to_numpy(dtype=float))) if len(group) else math.nan,
            "portfolio_sharpe": safe_sharpe(returns, context=f"{model} portfolio"),
            "portfolio_max_drawdown": safe_max_drawdown(returns, context=f"{model} portfolio"),
            "portfolio_turnover_mean": float(np.nanmean(group["turnover"].to_numpy(dtype=float))) if len(group) else math.nan,
        }
    return aggregates
