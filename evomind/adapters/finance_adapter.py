"""Finance adapter specialising in risk/credit prediction."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from evomind.adapters import register_task
from evomind.adapters.classification_adapter import ClassificationAdapter


@register_task("finance")
class FinanceAdapter(ClassificationAdapter):
    """Adds debt-to-income and utilization ratios before delegating to the base class."""

    def __init__(
        self,
        schema: Optional[Dict[str, object]] = None,
        default_target: str = "defaulted",
        *,
        data: Any | None = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(schema=schema, default_target=default_target, data=data, config=config)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if {"total_debt", "income"}.issubset(df.columns):
            df["debt_to_income"] = df["total_debt"] / df["income"].replace({0: 1})
        if {"credit_used", "credit_limit"}.issubset(df.columns):
            df["utilization"] = df["credit_used"] / df["credit_limit"].replace({0: 1})
        return super().preprocess(df)
