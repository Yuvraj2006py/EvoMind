"""
Metric utilities centralising scoring and fairness diagnostics.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn import metrics as skmetrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics used across adapters."""

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = skmetrics.mean_absolute_error(y_true, y_pred)
    mse = skmetrics.mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = skmetrics.r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))))

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": rmse,
        "r2_score": float(r2),
        "mape": mape,
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute standard classification metrics."""

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    precision = skmetrics.precision_score(y_true, y_pred, zero_division=0)
    recall = skmetrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = skmetrics.f1_score(y_true, y_pred, zero_division=0)
    accuracy = skmetrics.accuracy_score(y_true, y_pred)

    roc_auc = 0.0
    if y_proba is not None:
        y_proba = np.asarray(y_proba).reshape(-1)
        try:
            roc_auc = skmetrics.roc_auc_score(y_true, y_proba)
        except Exception:  # pragma: no cover - guard for degenerate cases
            roc_auc = 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "val_accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
    }


def demographic_parity(
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Compute demographic parity difference between sensitive groups."""

    y_pred = np.asarray(y_pred).reshape(-1)
    sensitive_attr = np.asarray(sensitive_attr).reshape(-1)
    unique_groups = np.unique(sensitive_attr)
    if len(unique_groups) < 2:
        return 0.0
    positive_rates = []
    for group in unique_groups:
        mask = sensitive_attr == group
        if mask.sum() == 0:
            continue
        positive_rates.append(float(np.mean(y_pred[mask] > 0.5)))
    if len(positive_rates) < 2:
        return 0.0
    return float(max(positive_rates) - min(positive_rates))


def equal_opportunity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """Compute equal opportunity difference (TPR gap) between sensitive groups."""

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    sensitive_attr = np.asarray(sensitive_attr).reshape(-1)

    unique_groups = np.unique(sensitive_attr)
    if len(unique_groups) < 2:
        return 0.0

    tpr_values = []
    for group in unique_groups:
        mask = sensitive_attr == group
        if mask.sum() == 0:
            continue
        tp = np.sum((y_true[mask] == 1) & (y_pred[mask] == 1))
        positives = np.sum(y_true[mask] == 1)
        if positives == 0:
            continue
        tpr_values.append(float(tp / positives))

    if len(tpr_values) < 2:
        return 0.0
    return float(max(tpr_values) - min(tpr_values))


def fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Wrapper combining demographic parity and equal opportunity metrics."""

    if sensitive_attr is None:
        return {"demographic_parity": 0.0, "equal_opportunity": 0.0}
    return {
        "demographic_parity": demographic_parity(y_pred, sensitive_attr),
        "equal_opportunity": equal_opportunity(y_true, y_pred, sensitive_attr),
    }
