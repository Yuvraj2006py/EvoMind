"""
Base interface for EvoMind task adapters.

Adapters are responsible for handling dataset specifics: loading, preprocessing,
building task aware models from genomes, evaluating them, and translating
metrics into a scalar fitness value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from torch import nn

from evomind.evolution.genome import Genome


class BaseTaskAdapter(ABC):
    """Abstract base class describing the adapter contract."""

    def __init__(self, schema: Optional[Dict[str, Any]] = None) -> None:
        """
        Parameters
        ----------
        schema : dict, optional
            Optional schema metadata inferred from the dataset profiler. Adapters
            can use this information to tailor preprocessing (e.g., type casting).
        """
        self._schema: Dict[str, Any] = {}
        self.schema = schema or {}

    @property
    def schema(self) -> Dict[str, Any]:
        return self._schema

    @schema.setter
    def schema(self, value: Optional[Dict[str, Any]]) -> None:
        self._schema = dict(value or {})

    @abstractmethod
    def load_data(self, path: Path) -> Tuple[Any, Any, Any, Any]:
        """
        Load and split a dataset.

        Returns
        -------
        Tuple[Any, Any, Any, Any]
            Training features, training targets, validation features, validation targets.
        """

    @abstractmethod
    def preprocess(self, X_train: Any, X_val: Any) -> Tuple[Any, Any]:
        """Apply preprocessing steps and return transformed training and validation features."""

    @abstractmethod
    def build_model(self, genome: Genome, input_dim: int, output_dim: int) -> nn.Module:
        """Create a PyTorch model instance for the provided genome description."""

    @abstractmethod
    def evaluate_model(self, model: nn.Module, X_val: Any, y_val: Any) -> Dict[str, float]:
        """Compute evaluation metrics for the trained model."""

    @abstractmethod
    def fitness(self, metrics: Dict[str, float]) -> float:
        """Translate task specific metrics into a scalar fitness value."""

    @abstractmethod
    def summarize_data(self, X_train: Any, y_train: Any) -> Dict[str, Any]:
        """Return dataset-level statistics used for insight generation."""

    @abstractmethod
    def generate_profile(self, X_train: Any, output_path: Path) -> Path:
        """Create a detailed dataset profiling report and return its path."""
