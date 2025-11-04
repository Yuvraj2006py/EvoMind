"""
Dataset profiling utilities used by EvoMind adapters.

The helper functions centralise lightweight summarisation logic so each adapter
can focus on data specific concerns while still producing a consistent insight
payload for the reporting pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="visions.utils.monkeypatches.imghdr_patch",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="matplotlib.cbook",
)

from .data_profiler import (
    compute_mutual_information,
    detect_constant_or_duplicate_columns,
    profile_health,
    time_series_diagnostics,
    to_serialisable,
)


def summarize_dataframe(df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate basic statistics for a dataframe.

    The summary contains column data types, missing value counts, and aggregate
    descriptive statistics.  All values are converted to plain Python types to
    make downstream serialisation straightforward.
    """

    summary: Dict[str, Any] = {}
    summary["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    summary["column_types"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    summary["missing_values"] = df.isnull().sum().astype(int).to_dict()
    summary["missing_pct"] = df.isnull().mean().fillna(0.0).to_dict()

    numeric_df = df.select_dtypes(include=[np.number])
    summary["skewness"] = {col: float(numeric_df[col].skew()) for col in numeric_df.columns}
    summary["kurtosis"] = {col: float(numeric_df[col].kurt()) for col in numeric_df.columns}

    try:
        describe = df.describe(include="all", datetime_is_numeric=True).transpose()
    except TypeError:
        describe = df.describe(include="all").transpose()
    describe_clean = describe.astype(object)
    describe_clean = describe_clean.where(pd.notna(describe_clean), None)
    summary["statistics"] = describe_clean.to_dict(orient="index")
    summary["correlation"] = df.corr(numeric_only=True).fillna(0.0).to_dict()

    # Data health and integrity warnings.
    column_health, health_score = profile_health(df)
    anomalies = detect_constant_or_duplicate_columns(df)
    high_corr = []
    corr_matrix = numeric_df.corr().abs().fillna(0.0)
    for col in corr_matrix.columns:
        for other in corr_matrix.columns:
            if col >= other:
                continue
            if corr_matrix.loc[col, other] > 0.95:
                high_corr.append({"features": [col, other], "correlation": float(corr_matrix.loc[col, other])})
    summary["health"] = {
        "score": health_score,
        "details": to_serialisable(column_health),
    }
    summary["integrity_warnings"] = {
        "constant": anomalies.get("constant", []),
        "duplicate": anomalies.get("duplicate", []),
        "high_correlation": high_corr,
    }

    # Mutual information ranking when target is known.
    if target and target in df.columns:
        task_type = "classification" if not np.issubdtype(df[target].dtype, np.number) else "regression"
        summary["mutual_information"] = compute_mutual_information(df, target, task_type)
    else:
        summary["mutual_information"] = []

    # Time-series diagnostics and decomposition.
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, tz]"]).columns
    ts_summary: Dict[str, Any] = {}
    if len(datetime_cols) > 0:
        dt_col = datetime_cols[0]
        candidate_target = target if target in df.columns else ""
        if not candidate_target:
            numeric_cols = [col for col in numeric_df.columns if col != dt_col]
            candidate_target = numeric_cols[0] if numeric_cols else ""
        if candidate_target:
            ts_summary = time_series_diagnostics(df, dt_col, candidate_target)
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose  # type: ignore[import]

                ts_df = df[[dt_col, candidate_target]].dropna().copy()
                ts_df[dt_col] = pd.to_datetime(ts_df[dt_col])
                ts_df = ts_df.sort_values(dt_col).set_index(dt_col)
                period = min(24, max(2, len(ts_df) // 6))
                decomposition = seasonal_decompose(ts_df[candidate_target], model="additive", period=period)
                ts_summary["trend"] = decomposition.trend.dropna().tolist()
                ts_summary["seasonal"] = decomposition.seasonal.dropna().tolist()
                ts_summary["residual"] = decomposition.resid.dropna().tolist()
            except Exception:  # pragma: no cover - optional dependency path
                ts_summary.setdefault("trend", [])
                ts_summary.setdefault("seasonal", [])
                ts_summary.setdefault("residual", [])
    summary["time_series"] = ts_summary

    try:
        summary["diversity_index"] = float(df.nunique().mean())
    except Exception:
        summary["diversity_index"] = 0.0
    return summary


def generate_profile_report(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Persist a dataset profiling HTML report.

    When ydata-profiling is available the richer profile is produced; otherwise a
    compact HTML summary derived from pandas statistics is written so the user
    still receives a useful artifact.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ydata_profiling import ProfileReport

        profile = ProfileReport(df, title="EvoMind Dataset Profile", minimal=True)
        profile.to_file(output_path)
    except Exception:  # pragma: no cover - optional dependency fallback path.
        try:
            html = df.describe(include="all", datetime_is_numeric=True).to_html()
        except TypeError:
            html = df.describe(include="all").to_html()
        output_path.write_text(
            "<html><head><title>EvoMind Dataset Profile</title></head><body>"
            "<h1>Dataset Summary</h1>"
            f"{html}"
            "</body></html>",
            encoding="utf-8",
        )
    return output_path
