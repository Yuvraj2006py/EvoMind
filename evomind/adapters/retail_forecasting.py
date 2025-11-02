"""
Retail forecasting adapter providing a concrete EvoMind task implementation.

The adapter consumes a CSV dataset describing store level sales and constructs
tabular features suitable for fully connected neural networks.  It intentionally
keeps preprocessing light to ensure the demo runs quickly while remaining easy
to extend with domain specific transformations later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from evomind.evolution.genome import Genome
from evomind.core.profiling import generate_profile_report, summarize_dataframe
from evomind.adapters.data_utils import create_one_hot_encoder, load_dataframe
from evomind.adapters import register_task
from evomind.adapters.base_adapter import BaseTaskAdapter


def _activation(name: str) -> nn.Module:
    """Map activation identifiers to torch modules."""
    mapping = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
    }
    return mapping.get(name, nn.ReLU())


@register_task("forecasting")
class RetailForecastingAdapter(BaseTaskAdapter):
    """Adapter implementing the EvoMind forecasting task."""

    def __init__(self, schema: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(schema=schema)
        self.scaler = StandardScaler()
        self.encoder = create_one_hot_encoder()
        self.task_type = "forecasting"
        self._latest_training_frame: pd.DataFrame | None = None
        self.feature_names_: list[str] = []
        self.target_column: Optional[str] = self.schema.get("target") if self.schema else None

    def load_data(self, path: Path) -> Tuple[Any, Any, Any, Any]:
        df = load_dataframe(path)
        target_column = self._identify_target_column(df)
        if target_column is None:
            raise ValueError("Dataset must contain a numeric sales column (e.g., 'sales' or 'final_amount').")
        self.target_column = target_column

        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
        if df[target_column].isnull().all():
            raise ValueError(f"Target column '{target_column}' must contain numeric values.")

        date_column = self._identify_date_column(df)
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            df["month"] = df[date_column].dt.month
            df["dayofweek"] = df[date_column].dt.dayofweek

        for col in df.columns:
            if col == date_column:
                continue
            if df[col].dtype == "object":
                converted = pd.to_numeric(df[col], errors="coerce")
                if not converted.isnull().all():
                    df[col] = converted

        features_to_drop: List[str] = [target_column]
        if date_column:
            features_to_drop.append(date_column)
        features = df.drop(columns=features_to_drop)
        target = df[target_column].astype(float)

        X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)
        self._latest_training_frame = X_train.copy()
        self._latest_training_frame[target_column] = y_train.values
        return X_train, y_train, X_val, y_val

    def _identify_target_column(self, df: pd.DataFrame) -> str | None:
        if self.target_column and self.target_column in df.columns:
            return self.target_column
        candidates = ["sales", "final_amount", "total_amount", "revenue"]
        for column in candidates:
            if column in df.columns:
                return column
        return None

    def _identify_date_column(self, df: pd.DataFrame) -> str | None:
        schema_dates = self.schema.get("datetime") if self.schema else []
        if schema_dates:
            for column in schema_dates:
                if column in df.columns:
                    return column
        candidates = ["date", "transaction_date", "timestamp", "datetime"]
        for column in candidates:
            if column in df.columns:
                return column
        return None

    def preprocess(self, X_train: Any, X_val: Any) -> Tuple[Any, Any]:
        cat_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
        num_cols = [col for col in X_train.columns if col not in cat_cols]

        X_train_num = self.scaler.fit_transform(X_train[num_cols])
        X_val_num = self.scaler.transform(X_val[num_cols])

        feature_names: list[str] = []
        if cat_cols:
            X_train_cat = self.encoder.fit_transform(X_train[cat_cols])
            X_val_cat = self.encoder.transform(X_val[cat_cols])
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
        layers = []
        in_dim = input_dim
        for layer in genome.layers:
            if layer.layer_type == "dense":
                units = int(layer.params["units"])
                layers.append(nn.Linear(in_dim, units))
                layers.append(_activation(layer.params.get("activation", "relu")))
                in_dim = units
            elif layer.layer_type == "dropout":
                layers.append(nn.Dropout(p=float(layer.params["p"])))
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def evaluate_model(self, model: nn.Module, X_val: Any, y_val: Any) -> Dict[str, float]:
        preds = model(X_val).detach()
        y_true = y_val.numpy().flatten()
        y_pred = preds.numpy().flatten()

        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-6, 1.0, np.abs(y_true))))

        threshold = np.maximum(np.abs(y_true), 1.0)
        within_tolerance = float(np.mean(np.abs(y_true - y_pred) < 0.1 * threshold))

        return {
            "val_loss": float(mse),
            "val_mae": float(mae),
            "rmse": float(rmse),
            "r2_score": float(r2),
            "mape": float(mape),
            "val_accuracy": within_tolerance,
            "robustness": float(1.0 - min(1.0, mae)),
        }

    def fitness(self, metrics: Dict[str, float]) -> float:
        # High accuracy and low loss produce higher scores.
        return metrics.get("val_accuracy", 0.0) - metrics.get("val_loss", 0.0)

    def summarize_data(self, X_train: Any, y_train: Any) -> Dict[str, Any]:
        if self._latest_training_frame is None:
            df = pd.DataFrame(X_train, columns=self.feature_names_ or None).assign(target=y_train)
        else:
            df = self._latest_training_frame.copy()
        return summarize_dataframe(df, target=self.target_column or 'target')

    def generate_profile(self, X_train: Any, output_path: Path) -> Path:
        if self._latest_training_frame is None:
            df = pd.DataFrame(X_train)
        else:
            df = self._latest_training_frame
        return generate_profile_report(df, output_path)
