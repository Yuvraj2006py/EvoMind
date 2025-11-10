"""Sports analytics adapter exposing performance trend features."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from evomind.adapters import register_task
from evomind.adapters.regression_adapter import RegressionAdapter


@register_task("sports")
class SportsAdapter(RegressionAdapter):
    """Model continuous performance targets such as efficiency ratings."""

    def __init__(
        self,
        schema: Optional[Dict[str, object]] = None,
        default_target: str = "efficiency",
        *,
        data: Any | None = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(schema=schema, default_target=default_target, data=data, config=config)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        rolling_cols = [col for col in df.columns if col.startswith("stat_")]
        for column in rolling_cols:
            df[f"{column}_trend"] = df[column].rolling(window=3, min_periods=1).mean()
        if "minutes_played" in df.columns:
            df["fatigue_index"] = df["minutes_played"].rolling(window=5, min_periods=1).mean()
        return super().preprocess(df)
