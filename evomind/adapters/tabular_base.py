"""
Shared utilities for tabular EvoMind adapters.

The `TabularAdapter` centralises preprocessing logic (scaling, encoding, data
splits) so concrete adapters only need to provide task specific evaluation and
fitness calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn

from evomind.evolution.genome import Genome
from evomind.core.profiling import generate_profile_report, summarize_dataframe
from evomind.adapters.base_adapter import BaseTaskAdapter
from evomind.adapters.data_utils import create_one_hot_encoder, load_dataframe


class TabularAdapter(BaseTaskAdapter):
    """Base class for tabular datasets with mixed numerical/categorical features."""

    def __init__(
        self,
        task_type: str,
        schema: Optional[Dict[str, Any]] = None,
        default_target: str = "target",
    ) -> None:
        self.default_target = default_target
        super().__init__(schema=schema)
        self.task_type = task_type
        self.target_column = self._resolve_schema_target(self.schema) or self.default_target
        self.scaler = StandardScaler()
        self.encoder = create_one_hot_encoder()
        self.label_encoder: LabelEncoder | None = None
        self.feature_names_: List[str] = []
        self._latest_training_frame: pd.DataFrame | None = None

    def _load_dataframe(self, path: Path) -> pd.DataFrame:
        return load_dataframe(path)

    def load_data(self, path: Path) -> Tuple[Any, Any, Any, Any]:
        df = self._load_dataframe(path)
        target_column = self._resolve_target_column(df)
        y = df[target_column]
        X = df.drop(columns=[target_column], errors="ignore")

        if self.task_type != "classification":
            y = self._coerce_numeric_target(y)
            valid_mask = ~y.isna()
            if valid_mask.sum() < 2:
                raise ValueError("Target column must contain at least two numeric values.")
            if not valid_mask.all():
                X = X.loc[valid_mask]
                y = y.loc[valid_mask]
            y = y.astype(np.float32)

        X = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            X[categorical_cols] = X[categorical_cols].fillna("missing")

        if self.task_type == "classification":
            y = y.astype(str)

        stratify = y if self.task_type == "classification" and y.nunique() > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        if self.task_type == "classification":
            self.label_encoder = LabelEncoder()
            y_train = pd.Series(self.label_encoder.fit_transform(y_train), index=y_train.index)
            y_val = pd.Series(self.label_encoder.transform(y_val), index=y_val.index)

        self._latest_training_frame = X_train.copy()
        self._latest_training_frame[target_column] = y_train.values
        self.target_column = target_column
        return X_train, y_train, X_val, y_val

    def _resolve_target_column(self, df: pd.DataFrame) -> str:
        if self.target_column and self.target_column in df.columns:
            return self.target_column
        fallback = df.columns[-1]
        self.target_column = fallback
        return fallback

    @staticmethod
    def _coerce_numeric_target(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float)
        as_str = series.astype(str).str.strip()
        cleaned = (
            as_str.str.replace(r"[^\d\-\.\,]", "", regex=True)
            .str.replace(",", "", regex=False)
            .replace({"": pd.NA, ".": pd.NA, "-": pd.NA})
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        return numeric

    def _resolve_schema_target(self, schema: Optional[Dict[str, Any]]) -> Optional[str]:
        if not schema:
            return None
        target = schema.get("target")
        return target if isinstance(target, str) and target else None

    @property
    def schema(self) -> Dict[str, Any]:  # type: ignore[override]
        return BaseTaskAdapter.schema.fget(self)  # type: ignore[attr-defined]

    @schema.setter
    def schema(self, value: Optional[Dict[str, Any]]) -> None:  # type: ignore[override]
        BaseTaskAdapter.schema.fset(self, value)  # type: ignore[attr-defined]
        target = self._resolve_schema_target(value)
        if target:
            self.target_column = target
        elif not getattr(self, "target_column", None):
            self.target_column = self.default_target

    def preprocess(self, X_train: Any, X_val: Any) -> Tuple[Any, Any]:
        X_train_df = pd.DataFrame(X_train)
        X_val_df = pd.DataFrame(X_val)
        cat_cols = [col for col in X_train_df.columns if X_train_df[col].dtype == "object"]
        num_cols = [col for col in X_train_df.columns if col not in cat_cols]

        X_train_num = self.scaler.fit_transform(X_train_df[num_cols])
        X_val_num = self.scaler.transform(X_val_df[num_cols])

        feature_names: List[str] = []
        if cat_cols:
            X_train_cat = self.encoder.fit_transform(X_train_df[cat_cols])
            X_val_cat = self.encoder.transform(X_val_df[cat_cols])
            X_train_prepared = np.hstack([X_train_num, X_train_cat])
            X_val_prepared = np.hstack([X_val_num, X_val_cat])
            feature_names = num_cols + list(self.encoder.get_feature_names_out(cat_cols))
        else:
            X_train_prepared = X_train_num
            X_val_prepared = X_val_num
            feature_names = num_cols

        self.feature_names_ = feature_names
        return X_train_prepared.astype(np.float32), X_val_prepared.astype(np.float32)

    def build_model(self, genome: Genome, input_dim: int, output_dim: int) -> nn.Module:
        layers: List[nn.Module] = []
        in_dim = input_dim
        for layer in genome.layers:
            if layer.layer_type == "dense":
                units = int(layer.params["units"])
                layers.append(nn.Linear(in_dim, units))
                activation_name = layer.params.get("activation", "relu")
                layers.append(self._activation(activation_name))
                in_dim = units
            elif layer.layer_type == "dropout":
                layers.append(nn.Dropout(p=float(layer.params["p"])))
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def evaluate_model(self, model: nn.Module, X_val: Any, y_val: Any) -> Dict[str, float]:
        raise NotImplementedError

    def fitness(self, metrics: Dict[str, float]) -> float:
        raise NotImplementedError

    def summarize_data(self, X_train: Any, y_train: Any) -> Dict[str, Any]:
        if self._latest_training_frame is None:
            df = pd.DataFrame(X_train)
            df[self.target_column] = y_train
        else:
            df = self._latest_training_frame.copy()
        return summarize_dataframe(df, target=getattr(self, 'target_column', None))

    def generate_profile(self, X_train: Any, output_path: Path) -> Path:
        if self._latest_training_frame is None:
            df = pd.DataFrame(X_train)
        else:
            df = self._latest_training_frame
        return generate_profile_report(df, output_path)

    @staticmethod
    def _activation(name: str) -> nn.Module:
        mapping = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.2),
        }
        return mapping.get(name, nn.ReLU())
