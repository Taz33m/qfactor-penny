from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qfactor_penny.metrics import safe_balanced_accuracy, safe_f1, safe_max_drawdown, safe_roc_auc, safe_sharpe, safe_spearman
from qfactor_penny.portfolio import aggregate_portfolio_metrics, build_portfolio_summary


def test_undefined_metrics_return_nan_with_warning():
    with pytest.warns(RuntimeWarning, match="ROC-AUC undefined"):
        assert math.isnan(safe_roc_auc(np.array([1, 1]), np.array([0.2, 0.3]), context="tiny"))
    with pytest.warns(RuntimeWarning, match="Spearman rank IC undefined"):
        assert math.isnan(safe_spearman(np.array([1.0, 1.0]), np.array([0.1, 0.2]), context="constant"))
    with pytest.warns(RuntimeWarning, match="Sharpe zero-volatility"):
        assert math.isnan(safe_sharpe(np.array([0.01, 0.01]), context="flat"))
    with pytest.warns(RuntimeWarning, match="Balanced accuracy undefined"):
        assert math.isnan(safe_balanced_accuracy(np.array([1, 1]), np.array([1, 1]), context="one-class"))
    with pytest.warns(RuntimeWarning, match="F1 undefined"):
        assert math.isnan(safe_f1(np.array([1, 1]), np.array([1, 1]), context="one-class"))
    with pytest.warns(RuntimeWarning, match="Max drawdown undefined"):
        assert math.isnan(safe_max_drawdown(np.array([np.nan]), context="empty"))


def test_portfolio_uses_absolute_returns_and_reports_alpha_vs_spy():
    rows = []
    tickers = ["XLB", "XLC", "XLE", "XLF"]
    for idx, ticker in enumerate(tickers):
        rows.append(
            {
                "split_id": "split_00",
                "date": "2024-01-02",
                "ticker": ticker,
                "model": "demo",
                "seed": 42,
                "score": 10 - idx,
                "forward_return_5d": 0.01 * (idx + 1),
                "spy_forward_return_5d": 0.005,
            }
        )
    predictions = pd.DataFrame(rows)
    portfolio = build_portfolio_summary(predictions, transaction_cost=0.0005)
    demo = portfolio[portfolio["model"] == "demo"].iloc[0]
    assert "SPY" not in demo["selected_tickers"]
    expected_gross = np.mean([0.01, 0.02, 0.03])
    assert demo["gross_return"] == pytest.approx(expected_gross)
    assert demo["turnover"] == pytest.approx(0.5)
    assert demo["transaction_cost"] == pytest.approx(0.00025)
    assert demo["net_return"] == pytest.approx(expected_gross - 0.00025)
    assert demo["alpha_vs_spy"] == pytest.approx(expected_gross - 0.00025 - 0.005)


def test_portfolio_ties_use_same_score_ticker_order_as_prediction_rank():
    rows = []
    for idx, ticker in enumerate(["XLY", "XLV", "XLB", "XLC", "XLE"]):
        rows.append(
            {
                "split_id": "split_00",
                "date": "2024-01-02",
                "ticker": ticker,
                "model": "demo",
                "seed": 42,
                "score": 1.0,
                "forward_return_5d": 0.01,
                "spy_forward_return_5d": 0.005,
            }
        )
    portfolio = build_portfolio_summary(pd.DataFrame(rows), transaction_cost=0.0005)
    demo = portfolio[portfolio["model"] == "demo"].iloc[0]
    assert demo["selected_tickers"] == "XLB,XLC,XLE"


def test_portfolio_turnover_resets_by_model_seed():
    rows = []
    for seed in [1, 2]:
        for date in ["2024-01-02", "2024-01-09"]:
            for idx, ticker in enumerate(["XLB", "XLC", "XLE", "XLF"]):
                rows.append(
                    {
                        "split_id": "split_00",
                        "seed": seed,
                        "date": date,
                        "ticker": ticker,
                        "model": "demo",
                        "score": 10 - idx,
                        "forward_return_5d": 0.01,
                        "spy_forward_return_5d": 0.005,
                    }
                )
    portfolio = build_portfolio_summary(pd.DataFrame(rows), transaction_cost=0.0005)
    demo = portfolio[portfolio["model"] == "demo"].sort_values(["seed", "date"])
    first_turnovers = demo.groupby("seed").head(1)["turnover"].to_numpy()
    assert np.allclose(first_turnovers, 0.5)
    aggregates = aggregate_portfolio_metrics(portfolio)
    assert ("demo", 1) in aggregates
    assert ("demo", 2) in aggregates
