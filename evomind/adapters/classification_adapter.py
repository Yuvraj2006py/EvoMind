"""
Tabular classification adapter implementation for EvoMind.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from evomind.adapters import register_task
from evomind.adapters.tabular_base import TabularAdapter


@register_task("classification")
class ClassificationAdapter(TabularAdapter):
    """
    Generic tabular classification adapter.

    The adapter expects a column named ``target`` containing class labels.
    Splits are stratified to maintain class balance.
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
            task_type="classification",
            schema=schema,
            default_target=default_target,
            data=data,
            config=config,
        )

    def train(self, X, y) -> LogisticRegression:
        model = LogisticRegression(max_iter=int(self.config.get("max_iter", 400)))
        model.fit(self._ensure_ndarray(X), self._ensure_ndarray(y).ravel())
        self.baseline_model_ = model
        return model

    def evaluate(self, model, X_val: Any, y_val: Any) -> Dict[str, float]:
        X_np = self._ensure_ndarray(X_val)
        y_np = self._ensure_ndarray(y_val)
        if y_np.ndim > 1:
            targets = np.argmax(y_np, axis=1)
        else:
            targets = y_np.astype(int).ravel()

        if hasattr(model, "predict_proba"):
            probs = np.asarray(model.predict_proba(X_np), dtype=np.float32)
            preds = probs.argmax(axis=1)
            logits = torch.from_numpy(probs)
        else:
            tensor = torch.as_tensor(X_np, dtype=torch.float32)
            with torch.no_grad():
                logits = model(tensor)
            if logits.ndim == 1 or logits.shape[1] == 1:
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
                preds = (probs.ravel() > 0.5).astype(int)
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)

        accuracy = float(accuracy_score(targets, preds))
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average="weighted", zero_division=0
        )

        roc_auc = None
        if probs.shape[1] == 2:
            roc_auc = roc_auc_score(targets, probs[:, 1])
        elif probs.shape[1] == 1:
            roc_auc = roc_auc_score(targets, probs[:, 0])

        logits_tensor = torch.from_numpy(probs) if isinstance(probs, np.ndarray) else logits
        targets_tensor = torch.as_tensor(targets, dtype=torch.long)
        if logits_tensor.ndim == 1 or logits_tensor.shape[1] == 1:
            loss = F.binary_cross_entropy(
                torch.clamp(logits_tensor.squeeze(), 1e-6, 1 - 1e-6),
                targets_tensor.float(),
            )
        else:
            loss = F.cross_entropy(torch.as_tensor(logits, dtype=torch.float32), targets_tensor)

        metrics = {
            "val_loss": float(loss.item()),
            "val_accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }
        if roc_auc is not None and not np.isnan(roc_auc):
            metrics["roc_auc"] = float(roc_auc)

        return metrics

    def fitness(self, metrics: Dict[str, float]) -> float:
        return (
            0.7 * metrics.get("val_accuracy", 0.0)
            + 0.3 * metrics.get("f1_score", 0.0)
            - 0.2 * metrics.get("val_loss", 0.0)
        )

    @staticmethod
    def _ensure_ndarray(data: Any) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return data
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        if hasattr(data, "to_numpy"):
            return data.to_numpy()
        return np.asarray(data)
