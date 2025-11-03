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

_TARGET_KEYWORDS = {
    "target": 3.0,
    "label": 2.5,
    "output": 2.0,
    "outcome": 2.0,
    "result": 2.0,
    "response": 1.5,
    "score": 1.2,
    "default": 1.2,
    "churn": 2.5,
    "fraud": 2.5,
    "sales": 2.5,
    "revenue": 2.5,
    "demand": 2.0,
    "price": 1.5,
    "forecast": 1.8,
    "yield": 1.2,
    "y": 0.8,
}


def _keyword_target(df: pd.DataFrame) -> tuple[str | None, float]:
    """Return the most likely target column based on keyword heuristics."""

    best_column: str | None = None
    best_score = 0.0
    total_rows = max(len(df), 1)

    for column in df.columns:
        lower = column.lower()
        score = 0.0
        for keyword, weight in _TARGET_KEYWORDS.items():
            if keyword in lower:
                score += weight

        if not score:
            continue

        uniques = df[column].nunique(dropna=False)
        unique_ratio = uniques / total_rows if total_rows else 0.0
        if pd.api.types.is_numeric_dtype(df[column]):
            if unique_ratio >= 0.2:
                score += 0.5
        else:
            if 0.01 < unique_ratio <= 0.3:
                score += 0.5

        if column == df.columns[-1]:
            score += 0.1

        if score > best_score:
            best_score = score
            best_column = column

    return best_column, best_score


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

    keyword_candidate, keyword_score = _keyword_target(df)
    if keyword_candidate and keyword_score >= 2.0:
        schema["target"] = keyword_candidate
        return schema

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
