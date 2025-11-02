"""
Fitness aggregation utilities for EvoMind evolution.

The evaluator transforms raw metrics coming from model evaluation into a single
scalar fitness score suitable for ranking genomes.  The default strategy uses a
weighted combination of accuracy, loss, latency, and robustness measures but it
can be extended or swapped for more advanced multi objective routines when we
integrate NSGA-II.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class FitnessEvaluator:
    """Compute a scalar fitness value from evaluation metrics."""

    accuracy_weight: float = 1.0
    loss_weight: float = 0.5
    latency_weight: float = 0.1
    robustness_weight: float = 0.2

    def score(self, metrics: Dict[str, float]) -> float:
        """
        Combine metrics into a fitness score.

        The calculation encourages high accuracy and low loss/latency.  A
        robustness score (higher is better) can also be supplied by adapters.
        """

        accuracy = metrics.get("val_accuracy", 0.0)
        loss = metrics.get("val_loss", 1.0)
        latency = metrics.get("latency", 0.5)
        robustness = metrics.get("robustness", 0.0)

        fitness = (
            self.accuracy_weight * accuracy
            - self.loss_weight * loss
            - self.latency_weight * latency
            + self.robustness_weight * robustness
        )
        return float(fitness)
