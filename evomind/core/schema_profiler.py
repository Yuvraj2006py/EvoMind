"""
Dataset schema profiling utilities for EvoMind.

The profiler inspects a pandas DataFrame and categorises columns by type while
attempting to infer the most likely prediction target.  The resulting schema
dictionary drives downstream task detection and adapter configuration.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def profile_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """Analyse a dataset and return a schema description."""

    schema: Dict[str, object] = {
        "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime": df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist(),
        "text": [],
        "target": None,
        "rows": len(df),
        "cols": len(df.columns),
    }

    # Heuristic detection of datetime columns from string representations.
    datetime_candidates = schema["datetime"]  # type: ignore[assignment]
    for column in df.columns:
        if column in datetime_candidates or df[column].dtype != "object":
            continue
        lowered = column.lower()
        if any(token in lowered for token in ["date", "time", "timestamp"]):
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().mean() > 0.5:
                datetime_candidates.append(column)
    schema["datetime"] = list(dict.fromkeys(datetime_candidates))  # type: ignore[assignment]

    # Identify text columns based on average string length.
    text_candidates = []
    for column in schema["categorical"]:  # type: ignore[index]
        avg_len = df[column].astype(str).str.len().mean()
        if pd.notna(avg_len) and avg_len > 50:
            text_candidates.append(column)
    schema["text"] = text_candidates

    schema["categorical"] = [  # type: ignore[index]
        column
        for column in schema["categorical"]
        if column not in schema["text"] and column not in schema["datetime"]  # type: ignore[index]
    ]

    # Infer potential target columns.
    numeric_columns = [c for c in schema["numeric"] if not df[c].isna().all()]  # type: ignore[index]
    if len(numeric_columns) == 1:
        schema["target"] = numeric_columns[0]
    elif "target" in df.columns:
        schema["target"] = "target"
    else:
        corr = df.corr(numeric_only=True)
        if not corr.empty:
            schema["target"] = corr.sum().abs().idxmax()
        else:
            schema["target"] = df.columns[-1]

    return schema
