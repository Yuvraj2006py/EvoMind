"""Built-in marketing adapter with campaign aware preprocessing."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from evomind.adapters import register_task
from evomind.adapters.classification_adapter import ClassificationAdapter


@register_task("marketing")
class MarketingAdapter(ClassificationAdapter):
    """Feature engineer campaign level metrics (spend ratios, recency, channels)."""

    def __init__(
        self,
        schema: Optional[Dict[str, object]] = None,
        default_target: str = "converted",
        *,
        data: Any | None = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(schema=schema, default_target=default_target, data=data, config=config)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "spend" in df.columns and "touches" in df.columns:
            df["spend_per_touch"] = df["spend"] / df["touches"].clip(lower=1)
        if "last_interaction" in df.columns:
            df["last_interaction"] = pd.to_datetime(df["last_interaction"], errors="coerce")
            reference = df["last_interaction"].max()
            df["days_since_last_interaction"] = (reference - df["last_interaction"]).dt.days.clip(lower=0)
            df = df.drop(columns=["last_interaction"])
        return super().preprocess(df)
