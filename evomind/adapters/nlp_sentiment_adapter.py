"""
NLP sentiment adapter leveraging TF-IDF features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

from evomind.adapters import register_task
from evomind.adapters.classification_adapter import ClassificationAdapter
from evomind.adapters.data_utils import load_dataframe


@register_task("nlp_sentiment")
class NLPSentimentAdapter(ClassificationAdapter):
    """Vectorises text inputs with TF-IDF and feeds them into dense networks."""

    def __init__(
        self,
        text_column: str = "text",
        label_column: str = "label",
        schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        schema = schema or {}
        text_cols = schema.get("text") or []
        resolved_text = text_cols[0] if text_cols else text_column
        target = schema.get("target") if isinstance(schema.get("target"), str) else None
        resolved_label = target or label_column
        super().__init__(schema=schema, default_target=resolved_label)
        self.text_column = resolved_text
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def _load_dataframe(self, path: Path) -> pd.DataFrame:  # type: ignore[override]
        df = load_dataframe(path)
        if self.text_column not in df.columns:
            raise ValueError(f"Dataset must contain a '{self.text_column}' column for text data.")
        if self.target_column not in df.columns:
            raise ValueError(f"Dataset must contain the target column '{self.target_column}'.")
        return df[[self.text_column, self.target_column]].rename(columns={self.text_column: "text"})

    def preprocess(self, X_train: Any, X_val: Any) -> Tuple[Any, Any]:  # type: ignore[override]
        train_series = pd.Series(X_train["text"]).astype(str)
        val_series = pd.Series(X_val["text"]).astype(str)
        X_train_vec = self.vectorizer.fit_transform(train_series)
        X_val_vec = self.vectorizer.transform(val_series)
        self.feature_names_ = list(self.vectorizer.get_feature_names_out())
        return X_train_vec.toarray().astype(np.float32), X_val_vec.toarray().astype(np.float32)
