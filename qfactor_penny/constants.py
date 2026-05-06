"""Shared constants for QFactor-Penny."""

from __future__ import annotations

BENCHMARK_TICKER = "SPY"

SECTOR_TICKERS = [
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_5d",
    "vol_20d",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "relative_strength_20d",
    "spy_vol_20d",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
]

REBALANCE_STEP_DAYS = 5
TOP_N = 3
BOTTOM_N = 3
