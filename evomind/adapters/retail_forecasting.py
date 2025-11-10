"""
Retail forecasting adapter providing seasonal feature engineering.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from evomind.adapters import register_task
from evomind.adapters.regression_adapter import RegressionAdapter


@register_task("retail")
@register_task("forecasting")
class RetailAdapter(RegressionAdapter):
    """Retail specific adapter with calendar features and anomaly cleaning."""

    task_type = "forecasting"

    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        default_target: str = "sales",
        *,
        data: Any | None = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(schema=schema, default_target=default_target, data=data, config=config)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        target = self._resolve_target_column(df)
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df[target] = df[target].rolling(window=2, min_periods=1).median()

        date_column = self._detect_date_column(df)
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            df["month"] = df[date_column].dt.month
            df["dayofweek"] = df[date_column].dt.dayofweek
            df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
            df = df.drop(columns=[date_column])

        promo_cols = [col for col in df.columns if "promo" in col.lower()]
        for col in promo_cols:
            df[col] = df[col].astype(int)

        return super().preprocess(df)

    def _detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        schema_dates = self.schema.get("datetime", [])
        for column in schema_dates or []:
            if column in df.columns:
                return column
        candidates = ["date", "transaction_date", "timestamp", "datetime"]
        for column in candidates:
            if column in df.columns:
                return column
        return None
