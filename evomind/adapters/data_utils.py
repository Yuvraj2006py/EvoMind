"""
Utility helpers for loading EvoMind datasets from various file formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_dataframe(path: Path, **read_kwargs: Any) -> pd.DataFrame:
    """
    Load a dataset into a pandas DataFrame supporting CSV and JSON inputs.

    Parameters
    ----------
    path : Path
        Location of the dataset file.
    read_kwargs : dict
        Optional keyword arguments forwarded to the pandas reader.
    """

    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return pd.read_csv(path, **read_kwargs)
    if suffix in {".json"}:
        try:
            return pd.read_json(path, **read_kwargs)
        except ValueError:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                payload = data.get("data", data)
                return pd.DataFrame(payload)
    raise ValueError(f"Unsupported dataset format for file: {path}")


def create_one_hot_encoder() -> OneHotEncoder:
    """
    Construct a OneHotEncoder compatible with scikit-learn versions >=1.2.

    The ``sparse`` keyword was replaced by ``sparse_output`` in newer releases.
    This helper abstracts the change to keep adapters concise.
    """

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - triggered on older scikit-learn versions.
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
