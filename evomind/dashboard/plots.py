"""
Plot generation helpers for the EvoMind dashboard.
"""

from __future__ import annotations

import plotly.express as px
import pandas as pd


def fitness_line_chart(df: pd.DataFrame):
    """Return a Plotly figure plotting best fitness per generation."""
    if df.empty:
        df = pd.DataFrame({"generation": [0], "best_fitness": [0.0]})
    fig = px.line(df, x="generation", y="best_fitness", title="Best Fitness")
    fig.update_traces(mode="lines+markers")
    return fig


def correlation_heatmap(corr: pd.DataFrame):
    """Return a heatmap visualising feature correlations."""
    if corr.empty:
        return px.imshow([[0]], labels=dict(x="feature", y="feature", color="corr"))
    fig = px.imshow(
        corr,
        labels=dict(color="Correlation"),
        x=corr.columns,
        y=corr.index,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(title="Feature Correlation Matrix")
    return fig


def feature_importance_bar(importance_df: pd.DataFrame):
    """Return a bar chart showing feature importance scores."""
    if importance_df.empty:
        importance_df = pd.DataFrame({"feature": ["n/a"], "importance": [0.0]})
    fig = px.bar(importance_df, x="feature", y="importance", title="Feature Importance")
    fig.update_layout(xaxis_title="Feature", yaxis_title="Importance")
    fig.update_traces(marker_color="#1f77b4")
    return fig


def pareto_front_scatter(df: pd.DataFrame, x_metric: str, y_metric: str):
    """Return a scatter plot approximating the Pareto front between two metrics."""
    if df.empty or x_metric not in df.columns or y_metric not in df.columns:
        return px.scatter(title="Pareto Front (insufficient data)")
    fig = px.scatter(
        df,
        x=x_metric,
        y=y_metric,
        color="generation" if "generation" in df.columns else None,
        title=f"Pareto Front: {x_metric} vs {y_metric}",
    )
    fig.update_traces(mode="markers", marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")))
    return fig


def mutual_info_bar(df: pd.DataFrame):
    """Return a bar chart ranking mutual information scores."""
    if df.empty or "feature" not in df.columns or "score" not in df.columns:
        return px.bar(title="Mutual Information (insufficient data)")
    df_sorted = df.sort_values("score", ascending=False)
    fig = px.bar(df_sorted, x="feature", y="score", title="Mutual Information")
    fig.update_layout(xaxis_title="Feature", yaxis_title="Score")
    return fig
