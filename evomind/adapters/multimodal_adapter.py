"""
Multimodal adapter combining text and numeric features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from evomind.adapters import register_task
from evomind.adapters.classification_adapter import ClassificationAdapter
from evomind.adapters.data_utils import load_dataframe


@register_task("multimodal")
class MultimodalAdapter(ClassificationAdapter):
    """Fuse text and numeric features into a single dense feature space."""

    def __init__(
        self,
        text_column: str = "text",
        label_column: str = "label",
        schema: Optional[Dict[str, Any]] = None,
        *,
        data: Any | None = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        schema = schema or {}
        text_cols = schema.get("text") or []
        resolved_text = text_cols[0] if text_cols else text_column
        target = schema.get("target") if isinstance(schema.get("target"), str) else None
        resolved_label = target or label_column
        super().__init__(schema=schema, default_target=resolved_label, data=data, config=config)
        self.text_column = resolved_text
        self.vectorizer = TfidfVectorizer(max_features=3000)
        self.scaler = StandardScaler()

    def _load_dataframe(self, path: Path) -> pd.DataFrame:  # type: ignore[override]
        df = load_dataframe(path)
        if self.text_column not in df.columns:
            raise ValueError(f"Dataset must contain a '{self.text_column}' column.")
        if self.target_column not in df.columns:
            raise ValueError(f"Dataset must contain the target column '{self.target_column}'.")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        df = df.copy()
        text = df[self.text_column].astype(str)
        tfidf = self.vectorizer.fit_transform(text).toarray().astype(np.float32)
        numeric_cols = [col for col in df.columns if col not in {self.text_column, self.target_column}]
        if numeric_cols:
            numeric_block = self.scaler.fit_transform(df[numeric_cols])
            combined = np.hstack([tfidf, numeric_block])
        else:
            combined = tfidf

        feature_names = list(self.vectorizer.get_feature_names_out()) + numeric_cols
        processed = pd.DataFrame(combined, columns=feature_names, index=df.index)
        processed[self.target_column] = df[self.target_column].values
        self.feature_names_ = feature_names
        return processed
