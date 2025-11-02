"""Generate lightweight textual summaries of EvoMind experiment insights."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


def _extract_top_features(feature_source: Optional[object], limit: int = 3) -> Iterable[str]:
    if feature_source is None:
        return []
    if isinstance(feature_source, Path):
        if not feature_source.exists():
            return []
        try:
            df = pd.read_csv(feature_source)
            if {"feature", "importance"}.issubset(df.columns):
                ordered = df.sort_values("importance", ascending=False)
                return ordered.head(limit)["feature"].tolist()
            return df.iloc[:, 0].head(limit).astype(str).tolist()
        except Exception:
            return []
    if isinstance(feature_source, Iterable):
        items = list(feature_source)
        return items[:limit]
    return []


def summarize_insights(
    metrics: Dict[str, float],
    feature_importance: Optional[object] = None,
    health_score: Optional[float] = None,
    anomalies: Optional[Dict[str, Iterable[str]]] = None,
    fairness: Optional[Dict[str, float]] = None,
) -> str:
    """Craft a concise narrative describing key outcomes of an EvoMind run."""

    parts = []
    r2 = metrics.get("r2_score")
    mae = metrics.get("val_mae") or metrics.get("mae")
    f1 = metrics.get("f1_score")
    accuracy = metrics.get("val_accuracy")
    stability = metrics.get("overfit_indicator")

    if r2 is not None:
        parts.append(f"Model achieved R^2 = {r2:.2f}.")
    if mae is not None:
        parts.append(f"Average absolute error {mae:.3f} across validation.")
    if f1 is not None:
        parts.append(f"Balanced classification F1 = {f1:.2f}.")
    if accuracy is not None and f1 is None:
        parts.append(f"Validation accuracy reached {accuracy:.2%}.")

    if stability is not None:
        if abs(stability) < 0.02:
            parts.append("Generalisation appears stable with minimal overfit.")
        elif stability > 0:
            parts.append("Slight overfitting detected; consider additional regularisation.")
        else:
            parts.append("Model may be underfitting, explore deeper architectures.")

    top_features = _extract_top_features(feature_importance)
    if top_features:
        parts.append("Top drivers: " + ", ".join(top_features) + ".")

    if health_score is not None:
        if health_score >= 85:
            parts.append(f"Data health scored {health_score:.0f}/100 with strong integrity.")
        elif health_score >= 70:
            parts.append(f"Data health scored {health_score:.0f}/100; review flagged issues for improvement.")
        else:
            parts.append(f"Data health scored {health_score:.0f}/100, remediation recommended.")

    if anomalies:
        alerts = []
        if anomalies.get("constant"):
            alerts.append("constant columns")
        if anomalies.get("duplicate"):
            alerts.append("duplicate columns")
        if anomalies.get("high_correlation"):
            alerts.append("high multicollinearity")
        if alerts:
            parts.append("Integrity warnings: " + ", ".join(alerts) + ".")

    if fairness:
        dp = fairness.get("demographic_parity")
        eo = fairness.get("equal_opportunity")
        if dp is not None or eo is not None:
            parts.append(
                "Fairness diagnostics - demographic parity gap {:.3f}, equal opportunity gap {:.3f}.".format(
                    float(dp or 0.0), float(eo or 0.0)
                )
            )

    if not parts:
        parts.append("Run completed successfully. Review detailed dashboards for deeper insights.")

    return " ".join(parts)
