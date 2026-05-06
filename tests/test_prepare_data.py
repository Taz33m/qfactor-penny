from __future__ import annotations

import numpy as np
import pandas as pd

from qfactor_penny.constants import SECTOR_TICKERS
from qfactor_penny.prepare_data import prepare_dataset


def test_prepare_data_excludes_spy_and_uses_non_overlapping_rebalance_dates(tmp_path):
    output = tmp_path / "qfactor_dataset.csv"
    frame = prepare_dataset(tmp_path / "missing_marketmind.csv", output)
    assert output.exists()
    assert "SPY" not in set(frame["ticker"])
    assert set(frame["ticker"]).issubset(set(SECTOR_TICKERS))
    dates = sorted(pd.to_datetime(frame["date"]).drop_duplicates())
    gaps = np.diff(np.array(dates, dtype="datetime64[D]")).astype("timedelta64[D]").astype(int)
    assert gaps.min() >= 5


def test_prepare_data_labels_top_3_bottom_3_and_drops_middle_from_training_label(tmp_path):
    output = tmp_path / "qfactor_dataset.csv"
    frame = prepare_dataset(tmp_path / "missing_marketmind.csv", output)
    first = frame[frame["date"] == frame["date"].min()].sort_values("excess_return_5d", ascending=False)
    assert "realized_rank_position" in frame.columns
    assert "rank_position" not in frame.columns
    assert first["realized_rank_position"].tolist() == list(range(1, len(first) + 1))
    assert first.head(3)["label"].eq(1.0).all()
    assert first.tail(3)["label"].eq(0.0).all()
    assert first.iloc[3:-3]["label"].isna().all()
