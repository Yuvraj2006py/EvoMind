"""
Lineage visualisation helpers for the EvoMind dashboard.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go


def build_lineage_figure(records: List[Dict]) -> Optional[go.Figure]:
    """
    Construct a Plotly scatter/line figure representing genome lineage.

    Parameters
    ----------
    records : list of dict
        Lineage records produced by the scheduler; each dict must contain
        generation, genome_id, fitness, and parent_ids.
    """

    if not records:
        return None

    df = pd.DataFrame(records)
    if df.empty or "generation" not in df or "genome_id" not in df:
        return None

    node_df = df.drop_duplicates(subset="genome_id")[["genome_id", "generation", "fitness"]]
    node_lookup = {
        row.genome_id: (row.generation, row.fitness) for row in node_df.itertuples(index=False)
    }

    fig = go.Figure()

    # Draw edges between parents and children.
    for row in df.itertuples(index=False):
        parents = row.parent_ids or []
        child_point = node_lookup.get(row.genome_id)
        if child_point is None:
            continue
        for parent_id in parents:
            parent_point = node_lookup.get(parent_id)
            if parent_point is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[parent_point[0], child_point[0]],
                    y=[parent_point[1], child_point[1]],
                    mode="lines",
                    line=dict(color="rgba(150,150,150,0.4)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.add_trace(
        go.Scatter(
            x=node_df["generation"],
            y=node_df["fitness"],
            mode="markers+text",
            text=[f"ID {gid}" for gid in node_df["genome_id"]],
            textposition="top center",
            marker=dict(size=10, color=node_df["generation"], colorscale="Blues", showscale=True),
            name="Genomes",
        )
    )

    fig.update_layout(
        title="Genome Lineage",
        xaxis_title="Generation",
        yaxis_title="Fitness",
        template="plotly_dark",
    )
    return fig

