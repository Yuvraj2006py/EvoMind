"""
Tabular classification adapter implementation for EvoMind.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch import nn

from evomind.adapters import register_task
from evomind.adapters.tabular_base import TabularAdapter


@register_task("classification")
class ClassificationAdapter(TabularAdapter):
    """
    Generic tabular classification adapter.

    The adapter expects a column named ``target`` containing class labels.
    Splits are stratified to maintain class balance.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None, default_target: str = "target") -> None:
        super().__init__(task_type="classification", schema=schema, default_target=default_target)

    def evaluate_model(self, model: nn.Module, X_val: Any, y_val: Any) -> Dict[str, float]:
        logits = model(X_val)
        probabilities = torch.softmax(logits, dim=1) if logits.shape[1] > 1 else torch.sigmoid(logits)
        probabilities_detached = probabilities.detach()
        predictions = (
            torch.argmax(probabilities_detached, dim=1)
            if probabilities_detached.ndim > 1
            else (probabilities_detached > 0.5).long().view(-1)
        )
        targets = torch.argmax(y_val, dim=1) if y_val.ndim > 1 else y_val.view(-1).long()

        accuracy = float(accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy()))
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets.cpu().numpy(),
            predictions.cpu().numpy(),
            average="weighted",
            zero_division=0,
        )

        roc_auc = None
        if probabilities_detached.ndim > 1 and probabilities_detached.shape[1] == 2:
            roc_auc = roc_auc_score(targets.cpu().numpy(), probabilities_detached[:, 1].cpu().numpy())
        elif probabilities_detached.ndim == 1:
            roc_auc = roc_auc_score(targets.cpu().numpy(), probabilities_detached.cpu().numpy())

        loss = F.binary_cross_entropy(probabilities, y_val.float()) if probabilities.ndim == 1 else F.cross_entropy(
            logits, targets
        )

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
