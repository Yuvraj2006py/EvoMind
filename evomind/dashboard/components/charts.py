"""
Reusable chart rendering helpers for the EvoMind dashboard.
"""

from __future__ import annotations

import streamlit as st


def render_line_chart(title: str, fig) -> None:
    """Render a Plotly line chart with a consistent layout."""
    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True)


def render_heatmap(title: str, fig) -> None:
    """Render a heatmap figure."""
    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True)


def render_bar_chart(title: str, fig) -> None:
    """Render a bar chart figure."""
    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True)


def render_scatter(title: str, fig) -> None:
    """Render a scatter chart figure."""
    st.subheader(title)
    st.plotly_chart(fig, use_container_width=True)
