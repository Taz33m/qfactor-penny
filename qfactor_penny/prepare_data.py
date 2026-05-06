"""Prepare non-overlapping sector rebalance data for QFactor-Penny."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import FEATURE_COLUMNS, REBALANCE_STEP_DAYS, SECTOR_TICKERS


REQUIRED_RETURN_COLUMNS = ["forward_return_5d", "spy_forward_return_5d", "excess_return_5d"]


def _synthetic_marketmind_frame(periods: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=periods)
    rows: list[dict[str, object]] = []
    for date_idx, date in enumerate(dates):
        spy_forward = 0.002 * np.sin(date_idx / 7.0)
        for ticker_idx, ticker in enumerate(SECTOR_TICKERS):
            phase = ticker_idx / 3.0
            alpha = 0.006 * np.sin(date_idx / 5.0 + phase) + 0.002 * np.cos(ticker_idx + date_idx / 11.0)
            forward_return = spy_forward + alpha
            base = date_idx + ticker_idx + 1
            row = {
                "date": date,
                "ticker": ticker,
                "close": 100.0 + base,
                "volume": 1_000_000 + 1000 * base,
                "forward_return_5d": forward_return,
                "spy_forward_return_5d": spy_forward,
                "excess_return_5d": forward_return - spy_forward,
            }
            row.update(
                {
                    "ret_1d": 0.002 * np.sin(base / 3.0),
                    "ret_5d": 0.004 * np.sin(base / 5.0),
                    "ret_20d": 0.010 * np.sin(base / 9.0),
                    "vol_5d": 0.01 + 0.002 * ((ticker_idx + date_idx) % 5),
                    "vol_20d": 0.015 + 0.001 * ((ticker_idx + 2 * date_idx) % 7),
                    "volume_ratio_5d": 0.8 + 0.05 * ((ticker_idx + date_idx) % 8),
                    "volume_ratio_20d": 0.9 + 0.04 * ((ticker_idx + 2 * date_idx) % 9),
                    "relative_strength_20d": 0.015 * np.sin(base / 8.0 + phase),
                    "spy_vol_20d": 0.012 + 0.002 * np.sin(date_idx / 13.0),
                    "weekday_sin": np.sin(2 * np.pi * date.weekday() / 5.0),
                    "weekday_cos": np.cos(2 * np.pi * date.weekday() / 5.0),
                    "month_sin": np.sin(2 * np.pi * date.month / 12.0),
                    "month_cos": np.cos(2 * np.pi * date.month / 12.0),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _load_input(input_path: Path) -> tuple[pd.DataFrame, bool, str]:
    if input_path.exists():
        return pd.read_csv(input_path), False, str(input_path)
    warnings.warn(
        f"Input dataset {input_path} was not found. Generating deterministic synthetic demo data.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _synthetic_marketmind_frame(), True, "synthetic"


def _add_forward_end_dates(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    unique_dates = sorted(pd.to_datetime(data["date"]).drop_duplicates())
    forward_end = {
        date: unique_dates[index + REBALANCE_STEP_DAYS] if index + REBALANCE_STEP_DAYS < len(unique_dates) else pd.NaT
        for index, date in enumerate(unique_dates)
    }
    data["forward_end_date"] = pd.to_datetime(data["date"]).map(forward_end)
    return data


def _label_rebalance_group(group: pd.DataFrame) -> pd.DataFrame:
    ranked = group.sort_values("excess_return_5d", ascending=False).copy()
    ranked["realized_rank_position"] = np.arange(1, len(ranked) + 1)
    ranked["label"] = np.nan
    if len(ranked) >= 6:
        ranked.loc[ranked.index[:3], "label"] = 1.0
        ranked.loc[ranked.index[-3:], "label"] = 0.0
    return ranked.sort_index()


def prepare_dataset(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)
    raw, synthetic, source = _load_input(input_path)
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["ticker"] = raw["ticker"].astype(str)
    data = raw[raw["ticker"].isin(SECTOR_TICKERS)].copy()
    missing = [column for column in [*FEATURE_COLUMNS, *REQUIRED_RETURN_COLUMNS] if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    data = _add_forward_end_dates(data)
    unique_dates = sorted(data["date"].drop_duplicates())
    rebalance_dates = set(unique_dates[::REBALANCE_STEP_DAYS])
    data = data[data["date"].isin(rebalance_dates)].copy()
    data = data.dropna(subset=["forward_end_date", *FEATURE_COLUMNS, *REQUIRED_RETURN_COLUMNS])
    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)
    data = pd.concat(
        [_label_rebalance_group(group) for _, group in data.groupby("date", sort=True)],
        ignore_index=True,
    )
    data["is_synthetic"] = bool(synthetic)
    data["data_source"] = source
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare QFactor-Penny rebalance data.")
    parser.add_argument("--input", required=True, help="MarketMind-Q frozen sector ETF CSV.")
    parser.add_argument("--output", required=True, help="Prepared QFactor-Penny CSV output.")
    args = parser.parse_args()
    frame = prepare_dataset(args.input, args.output)
    print(
        {
            "rows": int(len(frame)),
            "rebalance_dates": int(frame["date"].nunique()),
            "synthetic": bool(frame["is_synthetic"].any()),
            "output": args.output,
        }
    )


if __name__ == "__main__":
    main()
