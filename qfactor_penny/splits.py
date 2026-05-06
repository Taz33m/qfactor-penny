"""Expanding walk-forward split construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitSpec:
    split_id: str
    train_dates: list[pd.Timestamp]
    validation_dates: list[pd.Timestamp]
    test_dates: list[pd.Timestamp]
    purge_gap_days: int
    purge_trading_days_required: int
    train_purge_trading_days: int
    validation_purge_trading_days: int
    train_last_forward_end: pd.Timestamp
    validation_last_forward_end: pd.Timestamp

    def audit_row(self) -> dict[str, object]:
        return {
            "split_id": self.split_id,
            "train_start": _fmt(self.train_dates[0]),
            "train_end": _fmt(self.train_dates[-1]),
            "validation_start": _fmt(self.validation_dates[0]),
            "validation_end": _fmt(self.validation_dates[-1]),
            "test_start": _fmt(self.test_dates[0]),
            "test_end": _fmt(self.test_dates[-1]),
            "purge_gap_days": self.purge_gap_days,
            "purge_trading_days_required": self.purge_trading_days_required,
            "train_purge_trading_days": self.train_purge_trading_days,
            "validation_purge_trading_days": self.validation_purge_trading_days,
            "train_last_forward_end": _fmt(self.train_last_forward_end),
            "validation_last_forward_end": _fmt(self.validation_last_forward_end),
            "num_train_dates": len(self.train_dates),
            "num_validation_dates": len(self.validation_dates),
            "num_test_dates": len(self.test_dates),
        }


def _fmt(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _date_forward_ends(frame: pd.DataFrame) -> dict[pd.Timestamp, pd.Timestamp]:
    pairs = frame[["date", "forward_end_date"]].drop_duplicates()
    return {pd.Timestamp(row.date): pd.Timestamp(row.forward_end_date) for row in pairs.itertuples(index=False)}


def _business_day_gap(left: pd.Timestamp, right: pd.Timestamp) -> int:
    """Business days after left through right; used as the post-horizon purge gap."""
    left = pd.Timestamp(left)
    right = pd.Timestamp(right)
    if not left < right:
        return 0
    start = left + pd.offsets.BDay(1)
    if start > right:
        return 0
    return int(len(pd.bdate_range(start, right)))


def _clears_boundary(
    date: pd.Timestamp,
    *,
    boundary: pd.Timestamp,
    forward_ends: dict[pd.Timestamp, pd.Timestamp],
    purge_trading_days: int,
) -> bool:
    forward_end = forward_ends[date]
    return bool(forward_end < boundary and _business_day_gap(forward_end, boundary) >= purge_trading_days)


def make_walk_forward_splits(
    frame: pd.DataFrame,
    *,
    min_train_dates: int,
    validation_dates: int,
    purge_trading_days: int = 5,
    max_splits: int | None = None,
) -> tuple[list[SplitSpec], pd.DataFrame]:
    if purge_trading_days < 0:
        raise ValueError("purge_trading_days must be non-negative.")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["forward_end_date"] = pd.to_datetime(data["forward_end_date"])
    all_dates = sorted(data["date"].drop_duplicates())
    forward_ends = _date_forward_ends(data)
    periods = sorted(pd.Period(date, freq="M") for date in all_dates)
    unique_periods = []
    for period in periods:
        if not unique_periods or unique_periods[-1] != period:
            unique_periods.append(period)

    splits: list[SplitSpec] = []
    for period in unique_periods:
        test_dates = [date for date in all_dates if pd.Period(date, freq="M") == period]
        if not test_dates:
            continue
        test_start = test_dates[0]
        prior_dates = [date for date in all_dates if date < test_start]
        if len(prior_dates) < min_train_dates + validation_dates:
            continue

        validation_raw = prior_dates[-validation_dates:]
        validation_start = validation_raw[0]
        train_raw = [date for date in prior_dates if date < validation_start]
        train_dates = [
            date
            for date in train_raw
            if _clears_boundary(
                date,
                boundary=validation_start,
                forward_ends=forward_ends,
                purge_trading_days=purge_trading_days,
            )
        ]
        validation_clean = [
            date
            for date in validation_raw
            if _clears_boundary(
                date,
                boundary=test_start,
                forward_ends=forward_ends,
                purge_trading_days=purge_trading_days,
            )
        ]
        if len(train_dates) < min_train_dates or not validation_clean:
            continue

        train_last_forward_end = max(forward_ends[date] for date in train_dates)
        validation_last_forward_end = max(forward_ends[date] for date in validation_clean)
        train_gap = (validation_start - train_last_forward_end).days
        validation_gap = (test_start - validation_last_forward_end).days
        train_trading_gap = _business_day_gap(train_last_forward_end, validation_start)
        validation_trading_gap = _business_day_gap(validation_last_forward_end, test_start)
        split = SplitSpec(
            split_id=f"split_{len(splits):02d}_{period}",
            train_dates=train_dates,
            validation_dates=validation_clean,
            test_dates=test_dates,
            purge_gap_days=int(min(train_gap, validation_gap)),
            purge_trading_days_required=int(purge_trading_days),
            train_purge_trading_days=train_trading_gap,
            validation_purge_trading_days=validation_trading_gap,
            train_last_forward_end=train_last_forward_end,
            validation_last_forward_end=validation_last_forward_end,
        )
        splits.append(split)
        if max_splits is not None and len(splits) >= max_splits:
            break

    audit = pd.DataFrame([split.audit_row() for split in splits])
    return splits, audit
