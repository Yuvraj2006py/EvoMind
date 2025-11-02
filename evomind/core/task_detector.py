"""
Task detection utilities mapping dataset schemas to EvoMind adapters.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


def detect_task_type(df: pd.DataFrame, schema: Dict[str, object]) -> str:
    """
    Infer the most suitable adapter key for the provided dataset schema.

    Returns
    -------
    str
        Name of the registered adapter in ``TASK_REGISTRY``.
    """

    columns = df.columns.str.lower()

    if "image_path" in df.columns or any("img" in col for col in columns):
        return "vision"

    text_columns = schema.get("text", []) or []
    numeric_columns = schema.get("numeric", []) or []
    datetime_columns = schema.get("datetime", []) or []

    if text_columns:
        if numeric_columns:
            return "multimodal"
        return "nlp_sentiment"

    if datetime_columns:
        return "forecasting"

    target = schema.get("target")
    if isinstance(target, str) and target in df.columns:
        target_series = df[target]
        if target_series.dtype in ["object", "category"]:
            return "classification"
        if pd.api.types.is_integer_dtype(target_series) or pd.api.types.is_categorical_dtype(target_series):
            if target_series.nunique() < 15:
                return "classification"
        if pd.api.types.is_numeric_dtype(target_series):
            if target_series.nunique() <= 10:
                return "classification"
            return "regression"

    return "regression"
