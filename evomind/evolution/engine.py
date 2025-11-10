"""
Evolution engine orchestrating distributed, hybrid search strategies.

The engine now supports Ray powered parallel evaluation, Bayesian fine-tuning
with Optuna, and ensemble synthesis of high performing genomes. A fault-tolerant
execution layer guarantees graceful degradation to local execution when cluster
backends are unavailable.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
from dataclasses import dataclass, replace
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:  # pragma: no cover - optional dependency
    import optuna
except ImportError:  # pragma: no cover - optuna optional
    optuna = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import ray
except ImportError:  # pragma: no cover - ray optional
    ray = None  # type: ignore[assignment]

from concurrent.futures import ThreadPoolExecutor

from evomind.utils.logger import ExperimentLogger

from .fitness import FitnessEvaluator
from .genome import Genome
from .population import PopulationManager
from .search_space import SearchSpace
from .trainer import Trainer, TrainerConfig

if TYPE_CHECKING:
    from evomind.adapters.base_adapter import BaseTaskAdapter


def _ensure_ray_init() -> bool:
    """Initialise Ray only when available and not already started."""

    if ray is None:  # pragma: no cover - executed when ray missing
        return False
    if ray.is_initialized():  # pragma: no cover - quick exit branch
        return True
    try:  # pragma: no cover - defensive path
        ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
        return True
    except Exception:
        return False


def _evaluate_payload(payload: Dict[str, object]) -> Dict[str, object]:
    """Worker function used by both Ray and thread based execution."""

    genome: Genome = payload["genome"]  # type: ignore[assignment]
    adapter: "BaseTaskAdapter" = payload["adapter"]  # type: ignore[assignment]
    dataset = payload["dataset"]  # type: ignore[assignment]
    trainer_conf = payload["trainer_config"]  # type: ignore[assignment]
    logger_conf = payload["logger_conf"]  # type: ignore[assignment]

    trainer = Trainer(TrainerConfig(**trainer_conf), ExperimentLogger(**logger_conf))
    metrics, _ = trainer.train(genome, adapter, dataset)
    return {
        "genome_id": genome.id,
        "metrics": metrics,
    }


if ray is not None:  # pragma: no cover - only exercised when ray installed

    @ray.remote
    def _ray_evaluate(payload: Dict[str, object]) -> Dict[str, object]:
        return _evaluate_payload(payload)


@dataclass
class EvolutionConfig:
    """Hyperparameters guiding the hybrid evolutionary search."""

    elite_fraction: float = 0.2
    mutation_rate: float = 0.3
    crossover_rate: float = 0.6
    parallel_backend: str = "auto"  # auto|ray|threads
    bayes_rounds: int = 0
    ensemble_top_k: int = 3
    max_workers: Optional[int] = None


class ParallelExecutor:
    """Abstraction over Ray or local thread pools used for genome evaluation."""

    def __init__(self, backend: str, max_workers: Optional[int]) -> None:
        self._logger = logging.getLogger(__name__)
        backend = backend.lower()
        disable_ray = os.environ.get("EVOMIND_DISABLE_RAY", "").lower() in {"1", "true", "yes", "on"}
        on_windows = sys.platform.startswith("win")
        if backend == "auto":
            backend = "ray"
        if backend == "ray":
            if disable_ray:
                self._logger.info("EVOMIND_DISABLE_RAY detected. Using threaded execution.")
                backend = "threads"
            elif on_windows:
                self._logger.info("Ray disabled on Windows. Using threaded execution to avoid shared memory issues.")
                backend = "threads"
            elif not _ensure_ray_init():
                self._logger.info("Ray unavailable. Falling back to threaded execution.")
                backend = "threads"
        self.backend = backend if backend in {"ray", "threads"} else "threads"
        self.max_workers = max_workers
        self._last_stats: Dict[str, object] = {}

    def map(self, payloads: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        if self.backend == "ray" and ray is not None:
            try:
                handles = [_ray_evaluate.remote(payload) for payload in payloads]  # type: ignore[attr-defined]
                results = ray.get(handles)
                return list(results)
            except Exception as exc:  # pragma: no cover - defensive fallback
                self._logger.warning(
                    "Ray execution failed (%s). Falling back to threaded evaluation.",
                    exc,
                    exc_info=True,
                )
                self.backend = "threads"
        # Threaded fallback still provides parallel speedup without extra deps.
        workers = self.max_workers or min(8, max(1, len(payloads)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_evaluate_payload, payloads))

    def snapshot(self, payloads: int, duration: float) -> Dict[str, object]:
        stats: Dict[str, object] = {
            "backend": self.backend,
            "payloads": payloads,
            "duration": round(duration, 3),
            "max_workers": self.max_workers,
        }
        if self.backend == "threads":
            stats["worker_count"] = self.max_workers or min(8, max(1, payloads))
        if self.backend == "ray" and ray is not None and ray.is_initialized():  # pragma: no cover - ray optional
            try:
                stats["cluster_resources"] = {k: float(v) for k, v in ray.cluster_resources().items()}
                stats["available_resources"] = {k: float(v) for k, v in ray.available_resources().items()}
            except Exception:
                self._logger.debug("Unable to query Ray resources", exc_info=True)
        self._last_stats = stats
        return stats

    def last_stats(self) -> Dict[str, object]:
        return dict(self._last_stats)


class EvolutionEngine:
    """Central coordinator for the distributed EvoMind AutoML process."""

    def __init__(
        self,
        population: PopulationManager,
        trainer: Trainer,
        fitness_evaluator: FitnessEvaluator,
        search_space: SearchSpace,
        logger: ExperimentLogger,
        config: Optional[EvolutionConfig] = None,
    ) -> None:
        """Create a new evolutionary engine.

        Parameters
        ----------
        population : PopulationManager
            Initial population of genomes that will be evolved.
        trainer : Trainer
            Component responsible for fitting candidate models.
        fitness_evaluator : FitnessEvaluator
            Aggregates objective metrics into a scalar fitness score.
        search_space : SearchSpace
            Search space used to sample layers and genomes.
        logger : ExperimentLogger
            Logging utility for metrics and MLflow integration.
        config : EvolutionConfig, optional
            Hyper-parameters guiding selection, mutation, and parallelism.
        """
        self.population = population
        self.trainer = trainer
        self.fitness_evaluator = fitness_evaluator
        self.search_space = search_space
        self.logger = logger
        self.config = config or EvolutionConfig()
        self.executor = ParallelExecutor(self.config.parallel_backend, self.config.max_workers)
        self.ensemble_model: Optional[object] = None
        self.executor_stats_history: List[Dict[str, object]] = []

    def _build_payloads(
        self,
        adapter: "BaseTaskAdapter",
        dataset: Tuple,
    ) -> List[Dict[str, object]]:
        trainer_conf = {
            "epochs": self.trainer.config.epochs,
            "batch_size": self.trainer.config.batch_size,
            "learning_rate": self.trainer.config.learning_rate,
        }
        logger_conf = {
            "experiment_name": self.logger.experiment_name,
            "tracking_uri": self.logger.tracking_uri,
        }
        payloads = []
        for genome in self.population.genomes:
            payloads.append(
                {
                    "genome": genome.clone(),
                    "adapter": adapter,
                    "dataset": dataset,
                    "trainer_config": trainer_conf,
                    "logger_conf": logger_conf,
                }
            )
        return payloads

    def evaluate_generation(
        self,
        adapter: "BaseTaskAdapter",
        dataset: Tuple,
        generation_idx: int,
    ) -> List[Genome]:
        """Evaluate and rank genomes using the configured parallel backend.

        Parameters
        ----------
        adapter : BaseTaskAdapter
            Adapter supplying task-specific model construction and scoring.
        dataset : tuple
            Tuple containing training and validation tensors produced by the adapter.
        generation_idx : int
            Index of the generation currently being processed.

        Returns
        -------
        list[Genome]
            Genomes annotated with fitness, metrics, and generation metadata.
        """

        payloads = self._build_payloads(adapter, dataset)
        start_time = time.perf_counter()
        # Fault tolerance: re-run sequentially if the backend raises.
        with contextlib.suppress(Exception):
            results = self.executor.map(payloads)
        if "results" not in locals():  # pragma: no cover - fallback path
            self.logger.log_message("Parallel executor failed; falling back to sequential evaluation.")
            results = [_evaluate_payload(payload) for payload in payloads]

        duration = time.perf_counter() - start_time
        self.executor.snapshot(payloads=len(payloads), duration=duration)

        metrics_by_id = {item["genome_id"]: item["metrics"] for item in results}

        evaluated: List[Genome] = []
        for genome in self.population.genomes:
            metrics = metrics_by_id.get(genome.id, {})
            task_fitness = adapter.fitness(metrics)
            multi_obj_fitness = self.fitness_evaluator.score(metrics)
            genome.metrics = metrics
            genome.fitness = 0.5 * task_fitness + 0.5 * multi_obj_fitness
            genome.generation = generation_idx
            evaluated.append(genome)
            self.logger.log_metrics({"fitness": float(genome.fitness or 0.0)}, step=generation_idx)

        self.population.sort(key=lambda g: g.fitness or -math.inf)

        if self.config.bayes_rounds > 0:
            self._bayesian_refinement(adapter, dataset, evaluated[:3])

        return evaluated

    def _bayesian_refinement(
        self,
        adapter: "BaseTaskAdapter",
        dataset: Tuple,
        candidates: Sequence[Genome],
    ) -> None:
        """Run Optuna on top candidates to fine-tune trainer hyperparameters."""

        if optuna is None or not candidates:  # pragma: no cover - optional dep
            return

        base_config = self.trainer.config

        def objective(trial: "optuna.Trial") -> float:
            lr = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True)
            epochs = trial.suggest_int("epochs", max(3, base_config.epochs - 2), base_config.epochs + 3)
            cfg = replace(base_config, learning_rate=lr, epochs=epochs)
            trainer = Trainer(cfg, self.logger)
            genome = candidates[trial.number % len(candidates)].clone()
            metrics, _ = trainer.train(genome, adapter, dataset)
            score = self.fitness_evaluator.score(metrics)
            return float(score)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.bayes_rounds, show_progress_bar=False)
        best = study.best_params
        self.trainer.config.learning_rate = best.get("learning_rate", self.trainer.config.learning_rate)
        self.trainer.config.epochs = int(best.get("epochs", self.trainer.config.epochs))
        self.logger.log_message(
            f"Optuna refinement complete - lr={self.trainer.config.learning_rate:.3g}, "
            f"epochs={self.trainer.config.epochs}"
        )

    def _select_parents(self) -> List[Genome]:
        total = len(self.population.genomes)
        elite_count = max(1, int(total * self.config.elite_fraction))
        return list(self.population.top_k(elite_count))

    def _generate_offspring(self, parents: List[Genome], generation_idx: int) -> List[Genome]:
        offspring: List[Genome] = []
        if len(parents) < 2:
            return offspring

        for idx in range(0, len(parents) - 1, 2):
            parent_a, parent_b = parents[idx], parents[idx + 1]
            child = Genome.crossover(parent_a, parent_b)
            child = child.mutate(self.search_space, mutation_rate=self.config.mutation_rate)
            child.generation = generation_idx + 1
            offspring.append(child)
        return offspring

    def _synthesise_ensemble(self, adapter: "BaseTaskAdapter", dataset: Tuple, top_genomes: Sequence[Genome]) -> None:
        """Simple ensemble by averaging predictions from top genomes."""

        if not top_genomes or len(top_genomes) < 2:
            return

        X_train, y_train, X_val, y_val = dataset
        predictions = []
        for genome in top_genomes:
            metrics, model = self.trainer.train(genome.clone(), adapter, dataset)
            genome.metrics = metrics
            model.eval()
            with torch.no_grad(), np.errstate(all="ignore"):
                tensor = torch.as_tensor(X_val, dtype=torch.float32)
                preds = model(tensor).detach().cpu().numpy()
            predictions.append(preds)
        if predictions:
            self.ensemble_model = np.mean(predictions, axis=0)

    def evolve(self) -> None:
        """Trim the population to keep only the best genomes."""
        self.population.trim()

    def run_generation(
        self,
        adapter: "BaseTaskAdapter",
        dataset: Tuple,
        generation_idx: int,
    ) -> Tuple[Dict[str, float], Genome, List[Dict[str, object]], Dict[str, object]]:
        """Full generation loop: evaluation + reproduction + ensemble synthesis."""

        evaluated = self.evaluate_generation(adapter, dataset, generation_idx)
        parents = list(self._select_parents())
        offspring = self._generate_offspring(parents, generation_idx)

        next_generation: List[Genome] = []
        for parent in parents:
            child = parent.clone()
            child.generation = generation_idx + 1
            next_generation.append(child)
        next_generation.extend(offspring)
        while len(next_generation) < self.population.population_size:
            next_generation.append(
                Genome(layers=self.search_space.random_genome_spec(), generation=generation_idx + 1)
            )
        self.population.genomes = next_generation[: self.population.population_size]

        if self.config.ensemble_top_k > 1:
            self._synthesise_ensemble(adapter, dataset, evaluated[: self.config.ensemble_top_k])

        lineage: List[Dict[str, object]] = []
        for rank, genome in enumerate(evaluated[:10]):
            lineage.append(
                {
                    "generation": generation_idx,
                    "rank": rank,
                    "genome_id": genome.id,
                    "fitness": float(genome.fitness or 0.0),
                    "parent_ids": genome.parent_ids,
                    "metrics": genome.metrics,
                }
            )

        best_genome = evaluated[0] if evaluated else Genome(layers=[])
        snapshot = self.executor.last_stats()
        snapshot["generation"] = generation_idx
        self.executor_stats_history.append(snapshot)
        return (
            {
                "generation": generation_idx,
                "best_fitness": float(best_genome.fitness or 0.0),
            },
            best_genome,
            lineage,
            snapshot,
        )
