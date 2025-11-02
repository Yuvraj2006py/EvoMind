"""
Scheduler utilities orchestrating multiple generations of evolution.

This module provides an `EvolutionScheduler` class that keeps the high-level
experiment loop tidy and integrates experiment logging for every run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from evomind.evolution.genome import Genome
from evomind.utils.logger import ExperimentLogger
from evomind.adapters.base_adapter import BaseTaskAdapter

from .evolution import EvolutionEngine


@dataclass
class SchedulerConfig:
    """Configuration for the generation scheduler."""

    generations: int = 5
    run_name: str = "evomind-demo"
    log_history: bool = True


@dataclass
class EvolutionScheduler:
    """Drive the end-to-end evolution workflow."""

    engine: EvolutionEngine
    adapter: BaseTaskAdapter
    dataset: Tuple
    logger: ExperimentLogger
    config: SchedulerConfig
    history: List[Dict[str, float]] = field(default_factory=list)
    best_per_generation: List[Genome] = field(default_factory=list)
    lineage: List[Dict[str, object]] = field(default_factory=list)

    def run(self) -> List[Dict[str, float]]:
        """Execute the configured number of generations."""

        with self.logger.start_run(self.config.run_name):
            for generation in range(self.config.generations):
                metrics, best_genome, lineage = self.engine.run_generation(
                    self.adapter, self.dataset, generation_idx=generation
                )
                if self.config.log_history:
                    self.history.append(metrics)
                    self.best_per_generation.append(best_genome)
                    self.lineage.extend(lineage)
        return self.history
