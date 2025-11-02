"""
Tabular regression adapter for EvoMind.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from torch import nn

from evomind.adapters import register_task
from evomind.adapters.tabular_base import TabularAdapter


@register_task("regression")
class RegressionAdapter(TabularAdapter):
    """
    Generic regression adapter using dense networks.

    It expects the dataset to contain a numeric target column named ``target``.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None, default_target: str = "target") -> None:
        super().__init__(task_type="regression", schema=schema, default_target=default_target)

    def evaluate_model(self, model: nn.Module, X_val: Any, y_val: Any) -> Dict[str, float]:
        predictions = model(X_val).detach().view(-1)
        targets = y_val.view(-1)

        y_true = targets.cpu().numpy()
        y_pred = predictions.cpu().numpy()

        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-6, 1.0, np.abs(y_true))))

        return {
            "val_loss": float(mse),
            "val_mae": float(mae),
            "rmse": float(rmse),
            "r2_score": float(r2),
            "mape": float(mape),
        }

    def fitness(self, metrics: Dict[str, float]) -> float:
        return metrics.get("r2_score", 0.0) - 0.5 * metrics.get("val_mae", 0.0) - 0.5 * metrics.get("val_loss", 0.0)