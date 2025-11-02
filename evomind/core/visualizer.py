"""
Data preparation helpers for the EvoMind dashboard.

The dashboard expects tidy Pandas data frames to create figures.  This module
encapsulates that transformation so the Streamlit layer can focus on rendering.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, List


class EvolutionVisualizer:
    """Convert raw evolution history into dashboard friendly tables."""

    def __init__(self, history: List[Dict[str, float]]):
        self.history = history

    def to_dataframe(self) -> pd.DataFrame:
        """Return a pandas DataFrame summarising the evolution progress."""
        if not self.history:
            return pd.DataFrame(columns=["generation", "best_fitness"])
        return pd.DataFrame(self.history)

    def pareto_front(self) -> pd.DataFrame:
        """Placeholder for future Pareto front calculations."""
        df = self.to_dataframe()
        df["dominates"] = False
        return df
