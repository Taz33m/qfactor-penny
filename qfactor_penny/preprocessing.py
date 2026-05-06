"""Train-only feature selection and scaling."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .constants import FEATURE_COLUMNS


@dataclass
class FeaturePreprocessor:
    feature_count: int
    random_state: int = 42
    feature_selection_mode: str = "standard"
    min_cross_sectional_std_quantile: float = 0.25
    selected_features: list[str] | None = None
    feature_scores: dict[str, float] | None = None
    cross_sectional_std_by_feature: dict[str, float] | None = None
    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None

    def fit(self, train_frame: pd.DataFrame, y_train: np.ndarray) -> "FeaturePreprocessor":
        columns = FEATURE_COLUMNS
        self.imputer = SimpleImputer(strategy="median")
        raw = train_frame[columns].to_numpy(dtype=float)
        imputed = self.imputer.fit_transform(raw)
        scores = self._scores(imputed, y_train, random_state=self.random_state)
        self.cross_sectional_std_by_feature = self._cross_sectional_std(train_frame, columns)
        scores = self._apply_feature_selection_mode(scores, columns)
        self.feature_scores = {column: float(score) for column, score in zip(columns, scores)}
        selected_indices = np.argsort(scores)[::-1][: self.feature_count]
        selected_indices = np.sort(selected_indices)
        self.selected_features = [columns[index] for index in selected_indices]
        self.scaler = StandardScaler()
        self.scaler.fit(imputed[:, selected_indices])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.imputer is None or self.scaler is None or self.selected_features is None:
            raise RuntimeError("FeaturePreprocessor must be fit before transform.")
        raw = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        imputed = self.imputer.transform(raw)
        indices = [FEATURE_COLUMNS.index(name) for name in self.selected_features]
        return self.scaler.transform(imputed[:, indices])

    @staticmethod
    def _scores(x_train: np.ndarray, y_train: np.ndarray, *, random_state: int) -> np.ndarray:
        if len(np.unique(y_train)) < 2:
            warnings.warn("Feature selection received one-class labels; using variance scores.", RuntimeWarning)
            return np.var(x_train, axis=0)
        try:
            return mutual_info_classif(x_train, y_train, random_state=random_state)
        except Exception as exc:
            warnings.warn(f"Mutual information failed ({exc}); using variance scores.", RuntimeWarning)
            return np.var(x_train, axis=0)

    @staticmethod
    def _cross_sectional_std(train_frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        values: dict[str, float] = {}
        for column in columns:
            by_date = train_frame.groupby("date")[column].std(ddof=0)
            values[column] = float(by_date.mean()) if len(by_date) else 0.0
        return values

    def _apply_feature_selection_mode(self, scores: np.ndarray, columns: list[str]) -> np.ndarray:
        mode = self.feature_selection_mode or "standard"
        if mode == "standard":
            return scores
        if mode != "cross_sectional_aware":
            raise ValueError(f"Unknown feature selection mode: {mode}")
        if self.cross_sectional_std_by_feature is None:
            return scores
        cs_values = np.asarray([self.cross_sectional_std_by_feature[column] for column in columns], dtype=float)
        finite_positive = cs_values[np.isfinite(cs_values) & (cs_values > 0.0)]
        if len(finite_positive):
            threshold = float(np.nanquantile(finite_positive, self.min_cross_sectional_std_quantile))
        else:
            threshold = 0.0
        adjusted = np.asarray(scores, dtype=float).copy()
        keep = np.isfinite(cs_values) & (cs_values > 0.0) & (cs_values >= threshold)
        if int(np.sum(keep)) < self.feature_count:
            warnings.warn(
                "Cross-sectional-aware feature selection found fewer features than requested above the dispersion "
                "threshold; falling back to a penalized ranking instead of hard exclusion.",
                RuntimeWarning,
            )
            penalty = np.nanmax(np.abs(adjusted)) + 1.0 if len(adjusted) else 1.0
            adjusted = adjusted - np.where(keep, 0.0, penalty)
            return adjusted
        adjusted[~keep] = -np.inf
        return adjusted
