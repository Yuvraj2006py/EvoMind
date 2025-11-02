"""
Summary component highlighting key metrics from the latest EvoMind run.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


def render_summary(df: pd.DataFrame) -> None:
    """Render the headline statistics for the most recent experiment."""
    st.subheader("Experiment Summary")
    if df.empty:
        st.warning("No experiment history found. Run `python main.py ...` to generate results.")
        return

    best = df["best_fitness"].max()
    final = df.iloc[-1]["best_fitness"]
    st.metric("Best Fitness", f"{best:.4f}")
    st.metric("Final Generation Fitness", f"{final:.4f}")
