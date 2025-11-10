"""
Tabular regression adapter for EvoMind.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from evomind.adapters import register_task
from evomind.adapters.tabular_base import TabularAdapter


@register_task("regression")
class RegressionAdapter(TabularAdapter):
    """
    Generic regression adapter using dense networks.

    It expects the dataset to contain a numeric target column named ``target``.
    """

    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        default_target: str = "target",
        *,
        data: Any | None = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            task_type="regression",
            schema=schema,
            default_target=default_target,
            data=data,
            config=config,
        )

    # ---------------------------------------------------------------- Baseline
    def train(self, X, y) -> Ridge:
        model = Ridge(alpha=float(self.config.get("ridge_alpha", 1.0)))
        model.fit(self._ensure_ndarray(X), self._ensure_ndarray(y).ravel())
        self.baseline_model_ = model
        return model

    def evaluate(self, model, X_test, y_test) -> Dict[str, float]:
        X_np = self._ensure_ndarray(X_test)
        y_np = self._ensure_ndarray(y_test).ravel()
        if hasattr(model, "predict"):
            preds = np.asarray(model.predict(X_np), dtype=np.float32).ravel()
        else:
            tensor = torch.as_tensor(X_np, dtype=torch.float32)
            with torch.no_grad():
                preds = model(tensor).cpu().numpy().ravel()

        mse = mean_squared_error(y_np, preds)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_np, preds)
        r2 = r2_score(y_np, preds)
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.mean(np.abs((y_np - preds) / np.where(np.abs(y_np) < 1e-6, 1.0, np.abs(y_np))))

        return {
            "val_loss": float(mse),
            "val_mae": float(mae),
            "rmse": float(rmse),
            "r2_score": float(r2),
            "mape": float(mape),
        }

    def fitness(self, metrics: Dict[str, float]) -> float:
        return metrics.get("r2_score", 0.0) - 0.5 * metrics.get("val_mae", 0.0) - 0.5 * metrics.get("val_loss", 0.0)

    @staticmethod
    def _ensure_ndarray(data: Any) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return data
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        if hasattr(data, "to_numpy"):
            return data.to_numpy()
        return np.asarray(data)
