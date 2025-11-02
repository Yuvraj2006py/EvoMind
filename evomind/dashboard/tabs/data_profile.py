"""
Streamlit rendering utilities for the Data Profile tab.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from dashboard.plots import mutual_info_bar
from dashboard.components.charts import render_bar_chart


def render_data_profile_tab(
    profile: Dict,
    mutual_info_df: pd.DataFrame,
    anomalies: Dict,
    time_series_info: Dict,
) -> None:
    if not profile:
        st.info("Dataset profile not available. Run EvoMind with insights enabled.")
        return

    st.subheader("Dataset Health Overview")
    health_cols = st.columns(4)
    health_cols[0].metric("Rows", profile.get("rows", 0))
    health_cols[1].metric("Columns", profile.get("columns", 0))
    health_cols[2].metric("Health Score", f"{profile.get('health_score', 0):.1f}%")
    health_cols[3].metric("Missing Columns", profile.get("missing_columns", 0))

    st.markdown("### Column Diagnostics")
    detail_df = pd.DataFrame(profile.get("details", []))
    if detail_df.empty:
        st.info("Column diagnostics unavailable.")
    else:
        detail_df["missing_pct"] = detail_df["missing_pct"] * 100
        detail_df["outlier_pct"] = detail_df["outlier_pct"] * 100
        st.dataframe(detail_df.rename(columns={
            "missing_pct": "Missing %",
            "outlier_pct": "Outlier %",
            "skewness": "Skewness",
            "kurtosis": "Kurtosis",
        }))

    if anomalies:
        st.markdown("### Anomaly Detection")
        constant = anomalies.get("constant")
        duplicate = anomalies.get("duplicate")
        if constant:
            st.warning(f"Constant columns detected: {', '.join(constant)}")
        if duplicate:
            st.warning(f"Duplicate columns detected: {', '.join(duplicate)}")
        if not constant and not duplicate:
            st.success("No constant or duplicate columns detected.")

    st.markdown("### Mutual Information")
    if not mutual_info_df.empty:
        render_bar_chart("Mutual Information Ranking", mutual_info_bar(mutual_info_df.head(20)))
    else:
        st.info("Mutual information scores unavailable.")

    if time_series_info:
        st.markdown("### Time-series Diagnostics")
        rolling_mean = time_series_info.get("rolling_mean")
        rolling_var = time_series_info.get("rolling_var")
        if rolling_mean:
            st.line_chart(pd.Series(rolling_mean, name="Rolling Mean"))
        if rolling_var:
            st.line_chart(pd.Series(rolling_var, name="Rolling Variance"))
        if not rolling_mean and not rolling_var:
            st.info("Insufficient data for time-series decomposition.")

