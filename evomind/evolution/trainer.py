"""
Model training utilities used by EvoMind.

The trainer intentionally keeps the implementation lightweight so that the demo
can run quickly on CPU while still resembling an extensible training pipeline.
PyTorch Lightning integration is stubbed via a simple loop that mirrors the API
we will later expand upon.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from evomind.evolution.genome import Genome
from evomind.utils.logger import ExperimentLogger

if TYPE_CHECKING:
    from evomind.adapters.base_adapter import BaseTaskAdapter


@dataclass
class TrainerConfig:
    """Configuration block consumed by the trainer."""

    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3


class Trainer:
    """Simple torch based trainer with hooks into the EvoMind logging stack."""

    def __init__(self, config: TrainerConfig, logger: ExperimentLogger) -> None:
        self.config = config
        self.logger = logger

    def _prepare_dataloader(self, X: torch.Tensor, y: torch.Tensor) -> DataLoader:
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

    def train(
        self,
        genome: Genome,
        adapter: "BaseTaskAdapter",
        dataset: Tuple[Any, Any, Any, Any],
    ) -> Tuple[Dict[str, float], nn.Module]:
        """
        Train a model produced by the provided adapter.

        Returns the validation metrics dictionary supplied by the adapter with
        additional bookkeeping such as training loss and rough latency estimates.
        """

        X_train, y_train, X_val, y_val = dataset
        X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
        y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
        X_val_t = torch.as_tensor(X_val, dtype=torch.float32)
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32)

        if y_train_t.ndim == 1:
            y_train_t = y_train_t.unsqueeze(1)
        if y_val_t.ndim == 1:
            y_val_t = y_val_t.unsqueeze(1)

        model = adapter.build_model(genome, input_dim=X_train_t.shape[1], output_dim=y_train_t.shape[-1])

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        train_loader = self._prepare_dataloader(X_train_t, y_train_t)

        model.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            avg_loss = epoch_loss / len(train_loader.dataset)
            self.logger.log_message(f"Epoch {epoch+1}/{self.config.epochs} - loss={avg_loss:.4f}")

        # Validation pass
        model.eval()
        with torch.no_grad():
            start = time.perf_counter()
            preds = model(X_val_t)
            latency = time.perf_counter() - start

        metrics = adapter.evaluate_model(model, X_val_t, y_val_t)
        metrics.update(
            {
                "latency": latency,
                "train_loss": float(avg_loss),
            }
        )
        task_type = getattr(adapter, "task_type", "regression")
        self._augment_with_stability_metrics(
            adapter=adapter,
            model=model,
            task_type=task_type,
            X_train=X_train_t,
            y_train=y_train_t,
            X_val=X_val_t,
            y_val=y_val_t,
            preds=preds,
            metrics=metrics,
        )
        return metrics, model

    def _augment_with_stability_metrics(
        self,
        adapter: "BaseTaskAdapter",
        model: nn.Module,
        task_type: str,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        preds: torch.Tensor,
        metrics: Dict[str, float],
    ) -> None:
        """Compute additional diagnostics such as CV variance and overfitting."""

        primary_name, primary_val_score, higher_is_better = self._select_primary_metric(metrics, task_type)

        # Cross-validation style variance on the validation set.
        val_scores: List[float] = []
        n_samples = X_val.shape[0]
        splits = min(3, n_samples)
        if splits >= 2:
            kf = KFold(n_splits=splits, shuffle=True, random_state=42)
            for _, idx in kf.split(range(n_samples)):
                subset_X = X_val[idx]
                subset_y = y_val[idx]
                subset_metrics = adapter.evaluate_model(model, subset_X, subset_y)
                _, fold_score, _ = self._select_primary_metric(subset_metrics, task_type)
                if fold_score is not None:
                    val_scores.append(float(fold_score))
        if val_scores:
            metrics["cv_score_std"] = float(np.std(val_scores))

        # Overfit indicator.
        train_metrics = adapter.evaluate_model(model, X_train, y_train)
        _, train_score, _ = self._select_primary_metric(train_metrics, task_type)
        if train_score is not None and primary_val_score is not None:
            if higher_is_better:
                overfit = float(train_score - primary_val_score)
            else:
                overfit = float(primary_val_score - train_score)
            metrics["overfit_indicator"] = overfit
            metrics["train_primary_score"] = float(train_score)
            metrics["val_primary_score"] = float(primary_val_score)

        # Sensitivity drift via small perturbations.
        with torch.no_grad():
            perturbation = torch.randn_like(X_val) * 0.01
            perturbed_preds = model(X_val + perturbation)
            drift = torch.mean(torch.abs(perturbed_preds - preds))
        metrics["sensitivity_drift"] = float(drift.cpu().item())

    def _select_primary_metric(
        self,
        metrics: Dict[str, float],
        task_type: str,
    ) -> Tuple[str, Optional[float], bool]:
        """Choose a metric that best represents performance for stability checks."""

        task_type = task_type or "regression"
        priorities = {
            "classification": [
                ("f1_score", True),
                ("val_accuracy", True),
                ("precision", True),
                ("recall", True),
                ("roc_auc", True),
                ("val_loss", False),
            ],
            "forecasting": [
                ("r2_score", True),
                ("rmse", False),
                ("val_loss", False),
                ("val_mae", False),
                ("mape", False),
            ],
            "regression": [
                ("r2_score", True),
                ("rmse", False),
                ("val_loss", False),
                ("val_mae", False),
                ("mape", False),
            ],
        }
        for name, higher in priorities.get(task_type, []) + priorities["regression"]:
            if name in metrics and metrics[name] is not None:
                return name, float(metrics[name]), higher
        return "", None, True
