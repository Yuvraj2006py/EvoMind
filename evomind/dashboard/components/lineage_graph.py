"""
Placeholder component for the EvoMind architecture lineage visualisation.

Future revisions will leverage NetworkX and Graphviz to display ancestor
relationships between genomes.  For now the component renders a gentle reminder
that the feature is coming soon.
"""

from __future__ import annotations

import streamlit as st


def render_lineage_placeholder() -> None:
    """Render placeholder content for the lineage graph."""
    st.subheader("Architecture Lineage")
    st.info(
        "Lineage visualisation is under construction. "
        "Future releases will display parent-child relationships between genomes."
    )
