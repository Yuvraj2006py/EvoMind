"""
Multimodal adapter combining text and numeric features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    ) -> None:
        schema = schema or {}
        text_cols = schema.get("text") or []
        resolved_text = text_cols[0] if text_cols else text_column
        target = schema.get("target") if isinstance(schema.get("target"), str) else None
        resolved_label = target or label_column
        super().__init__(schema=schema, default_target=resolved_label)
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

    def preprocess(self, X_train: Any, X_val: Any) -> Tuple[Any, Any]:  # type: ignore[override]
        X_train_df = pd.DataFrame(X_train)
        X_val_df = pd.DataFrame(X_val)
        text_train = X_train_df[self.text_column].astype(str)
        text_val = X_val_df[self.text_column].astype(str)
        X_train_vec = self.vectorizer.fit_transform(text_train)
        X_val_vec = self.vectorizer.transform(text_val)

        numeric_cols = [col for col in X_train_df.columns if col not in {self.text_column, self.target_column}]
        if numeric_cols:
            X_train_num = self.scaler.fit_transform(X_train_df[numeric_cols])
            X_val_num = self.scaler.transform(X_val_df[numeric_cols])
        else:
            X_train_num = np.empty((len(X_train_df), 0))
            X_val_num = np.empty((len(X_val_df), 0))

        X_train_combined = np.hstack([X_train_vec.toarray(), X_train_num])
        X_val_combined = np.hstack([X_val_vec.toarray(), X_val_num])
        text_features = list(self.vectorizer.get_feature_names_out())
        self.feature_names_ = text_features + numeric_cols
        return X_train_combined.astype(np.float32), X_val_combined.astype(np.float32)
