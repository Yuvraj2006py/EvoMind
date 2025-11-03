"""
Generic preprocessing fallback used when specialised adapters fail.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def generic_preprocess(df: pd.DataFrame, target: str | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply a simple preprocessing pipeline to an arbitrary dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to preprocess.
    target : str, optional
        Column to treat as the prediction target.  When omitted the final column is used.
    """

    df = df.copy()
    if target is None or target not in df.columns:
        target = df.columns[-1]

    y = df[target]
    if y.dtype == "object":
        as_str = y.astype(str).str.strip()
        cleaned = (
            as_str.str.replace(r"[^\d\-\.\,]", "", regex=True)
            .str.replace(",", "", regex=False)
            .replace({"": pd.NA, ".": pd.NA, "-": pd.NA})
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().sum() >= len(y) * 0.5:
            y = numeric
        else:
            y = as_str.fillna("missing")
    else:
        y = y.copy()
    X = df.drop(columns=[target], errors="ignore")

    for column in X.columns:
        if X[column].dtype == "object":
            encoder = LabelEncoder()
            X[column] = encoder.fit_transform(X[column].astype(str).fillna("missing"))

    X = X.fillna(0.0)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled, y
