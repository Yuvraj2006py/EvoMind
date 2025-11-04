"""
Extended dataset profiling utilities used for EvoMind insights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


@dataclass
class ColumnHealth:
    column: str
    missing_pct: float
    outlier_pct: float
    skewness: float | None
    kurtosis: float | None


def _compute_outlier_ratio(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    series = series.dropna()
    if series.empty:
        return 0.0
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return float(mask.mean())


def profile_health(df: pd.DataFrame) -> Tuple[List[ColumnHealth], float]:
    """Return health metrics per column and an aggregated health score."""

    details: List[ColumnHealth] = []
    health_scores: List[float] = []

    for column in df.columns:
        series = df[column]
        missing_pct = float(series.isna().mean())
        skewness = float(series.skew()) if np.issubdtype(series.dropna().dtype, np.number) else None
        kurtosis = float(series.kurt()) if np.issubdtype(series.dropna().dtype, np.number) else None
        outlier_pct = _compute_outlier_ratio(series.astype(float)) if np.issubdtype(series.dropna().dtype, np.number) else 0.0

        details.append(
            ColumnHealth(
                column=column,
                missing_pct=missing_pct,
                outlier_pct=outlier_pct,
                skewness=skewness,
                kurtosis=kurtosis,
            )
        )

        column_health = 1.0 - min(1.0, missing_pct + outlier_pct)
        health_scores.append(column_health)

    health_score = float(np.mean(health_scores) * 100) if health_scores else 100.0
    return details, health_score


def detect_constant_or_duplicate_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    constant = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    duplicate = []
    seen = {}
    for col in df.columns:
        series_tuple = tuple(df[col].fillna("__nan__"))
        if series_tuple in seen:
            duplicate.append(col)
        else:
            seen[series_tuple] = col
    return {"constant": constant, "duplicate": duplicate}


def compute_mutual_information(
    df: pd.DataFrame,
    target: str,
    task_type: str,
) -> List[Dict[str, float]]:
    """Compute mutual information scores for features relative to the target."""

    if target not in df.columns:
        return []

    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    y = df[target]

    if X.empty or y.isna().all():
        return []

    # Guard against invalid values.
    X = X.dropna(axis=1, how="all")
    if X.empty:
        return []
    X = X.fillna(X.mean())
    if X.isna().any().any():
        X = X.fillna(0.0)

    if task_type == "classification":
        mode = y.mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "missing"
        y_filled = y.fillna(fill_value)
        encoded, uniques = pd.factorize(y_filled)
        if len(uniques) <= 1:
            return []
        y_values = encoded
    else:
        if np.issubdtype(y.dtype, np.number):
            y_numeric = y.astype(float)
        else:
            y_numeric = pd.to_numeric(y, errors="coerce")
        if y_numeric.isna().all():
            return []
        y_values = y_numeric.fillna(y_numeric.mean())

    try:
        if task_type == "classification":
            scores = mutual_info_classif(X, y_values)
        else:
            scores = mutual_info_regression(X, y_values)
    except Exception:
        return []

    ranking = sorted(zip(X.columns, scores), key=lambda item: item[1], reverse=True)
    return [{"feature": feature, "score": float(score)} for feature, score in ranking]


def time_series_diagnostics(df: pd.DataFrame, datetime_column: str, target: str) -> Dict[str, List[float]]:
    """
    Provide lightweight time-series diagnostics (rolling stats, autocorrelation).

    Uses pandas to compute rolling mean/variance and statsmodels if available.
    """

    diagnostics: Dict[str, List[float]] = {}
    if datetime_column not in df.columns or target not in df.columns:
        return diagnostics

    ts_df = df[[datetime_column, target]].dropna().copy()
    if ts_df.empty:
        return diagnostics

    ts_df[datetime_column] = pd.to_datetime(ts_df[datetime_column], errors="coerce")
    ts_df = ts_df.dropna().sort_values(datetime_column)

    rolling = ts_df[target].rolling(window=min(30, len(ts_df))).agg(["mean", "var"])
    diagnostics["rolling_mean"] = rolling["mean"].bfill().tolist()
    diagnostics["rolling_var"] = rolling["var"].bfill().tolist()

    try:  # optional statsmodels
        from statsmodels.tsa.stattools import acf  # type: ignore[import]

        acf_values = acf(ts_df[target], nlags=min(20, len(ts_df) - 1), fft=False)
        diagnostics["acf"] = acf_values.tolist()
    except Exception:  # pragma: no cover
        diagnostics["acf"] = []

    return diagnostics


def to_serialisable(details: List[ColumnHealth]) -> List[Dict[str, float]]:
    """Convert dataclass metrics into dictionaries suitable for JSON serialization."""
    return [asdict(row) for row in details]
