from __future__ import annotations

import pandas as pd

from qfactor_penny.constants import FEATURE_COLUMNS, SECTOR_TICKERS
from qfactor_penny.prepare_data import prepare_dataset
from qfactor_penny.preprocessing import FeaturePreprocessor
from qfactor_penny.splits import make_walk_forward_splits


def test_split_audit_columns_and_forward_windows_do_not_overlap(tmp_path):
    frame = prepare_dataset(tmp_path / "missing.csv", tmp_path / "data.csv")
    splits, audit = make_walk_forward_splits(frame, min_train_dates=8, validation_dates=3, purge_trading_days=5, max_splits=2)
    required = {
        "split_id",
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
        "purge_gap_days",
        "purge_trading_days_required",
        "train_purge_trading_days",
        "validation_purge_trading_days",
        "train_last_forward_end",
        "validation_last_forward_end",
        "num_train_dates",
        "num_validation_dates",
        "num_test_dates",
    }
    assert required.issubset(set(audit.columns))
    assert splits
    forward_ends = frame[["date", "forward_end_date"]].drop_duplicates()
    forward_map = {pd.Timestamp(row.date): pd.Timestamp(row.forward_end_date) for row in forward_ends.itertuples(index=False)}
    for split in splits:
        validation_start = split.validation_dates[0]
        test_start = split.test_dates[0]
        assert not (set(split.train_dates) & set(split.validation_dates))
        assert not (set(split.train_dates) & set(split.test_dates))
        assert not (set(split.validation_dates) & set(split.test_dates))
        assert max(split.train_dates) < validation_start < min(split.test_dates)
        assert all(forward_map[date] < validation_start for date in split.train_dates)
        assert all(forward_map[date] < test_start for date in split.validation_dates)
        assert all(forward_map[date] < test_start for date in split.train_dates)
        assert split.purge_gap_days > 0
        assert split.train_purge_trading_days >= split.purge_trading_days_required
        assert split.validation_purge_trading_days >= split.purge_trading_days_required


def test_purge_trading_days_is_enforced(tmp_path):
    frame = prepare_dataset(tmp_path / "missing.csv", tmp_path / "data.csv")
    splits, audit = make_walk_forward_splits(frame, min_train_dates=8, validation_dates=3, purge_trading_days=10, max_splits=2)
    assert splits
    assert audit["purge_trading_days_required"].eq(10).all()
    assert audit["train_purge_trading_days"].ge(10).all()
    assert audit["validation_purge_trading_days"].ge(10).all()


def test_preprocessor_fits_train_only_and_selects_requested_features(tmp_path):
    frame = prepare_dataset(tmp_path / "missing.csv", tmp_path / "data.csv")
    splits, _ = make_walk_forward_splits(frame, min_train_dates=8, validation_dates=3, purge_trading_days=5, max_splits=1)
    train = frame[frame["date"].isin(splits[0].train_dates)]
    labeled = train[train["label"].notna()]
    preprocessor = FeaturePreprocessor(feature_count=4).fit(labeled, labeled["label"].to_numpy(dtype=int))
    transformed = preprocessor.transform(labeled)
    assert len(preprocessor.selected_features or []) == 4
    assert transformed.shape[1] == 4
    assert preprocessor.imputer is not None
    assert preprocessor.scaler is not None


def test_cross_sectional_aware_feature_selection_uses_training_dispersion_only():
    dates = pd.bdate_range("2024-01-02", periods=8)
    rows = []
    for date_idx, date in enumerate(dates):
        for ticker_idx, ticker in enumerate(SECTOR_TICKERS):
            row = {"date": date, "ticker": ticker}
            for feature in FEATURE_COLUMNS:
                row[feature] = float(ticker_idx + date_idx / 10.0)
            row["weekday_cos"] = float(date_idx)
            row["month_sin"] = float(date_idx)
            row["month_cos"] = float(date_idx)
            row["spy_vol_20d"] = float(date_idx)
            rows.append(row)
    frame = pd.DataFrame(rows)
    y = (frame["ticker"].isin(SECTOR_TICKERS[:6])).astype(int).to_numpy()
    preprocessor = FeaturePreprocessor(
        feature_count=4,
        feature_selection_mode="cross_sectional_aware",
        min_cross_sectional_std_quantile=0.25,
    ).fit(frame, y)
    cross = preprocessor.cross_sectional_std_by_feature or {}
    positive_cross = pd.Series(cross)[lambda values: values > 0.0]
    threshold = positive_cross.quantile(0.25)
    assert preprocessor.selected_features
    assert all(cross[feature] >= threshold for feature in preprocessor.selected_features)
    assert not {"weekday_cos", "month_sin", "month_cos", "spy_vol_20d"} & set(preprocessor.selected_features)


def test_cross_sectional_aware_feature_selection_excludes_zero_dispersion_when_quantile_is_zero():
    preprocessor = FeaturePreprocessor(
        feature_count=4,
        feature_selection_mode="cross_sectional_aware",
        min_cross_sectional_std_quantile=0.25,
    )
    dispersed = set(FEATURE_COLUMNS[:4])
    preprocessor.cross_sectional_std_by_feature = {
        feature: (1.0 if feature in dispersed else 0.0) for feature in FEATURE_COLUMNS
    }
    scores = pd.Series(10.0, index=FEATURE_COLUMNS)
    adjusted = preprocessor._apply_feature_selection_mode(scores.to_numpy(dtype=float), FEATURE_COLUMNS)
    selected = {FEATURE_COLUMNS[index] for index in adjusted.argsort()[::-1][:4]}
    assert selected == dispersed
