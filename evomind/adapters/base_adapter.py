"""Modernised adapter base class powering EvoMind's extensibility."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from torch import nn

from evomind.core.profiling import generate_profile_report, summarize_dataframe
from evomind.evolution.genome import Genome
from evomind.exceptions import EvoMindAdapterError

DataSource = Union[str, Path, pd.DataFrame]


class BaseAdapter(ABC):
    """
    Base class for all EvoMind domain adapters.

    Concrete adapters encapsulate domain specific I/O, preprocessing, and
    baseline model training/evaluation without touching the core AutoML engine.
    Subclasses may override :meth:`build_model` when they need custom neural
    network topologies, otherwise the default dense builder will be used.
    """

    task_type: str = "regression"
    baseline_model_: Any
    feature_names_: list[str]
    target_column: Optional[str]

    def __init__(
        self,
        *,
        schema: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        target_col: Optional[str] = None,
        data: Optional[DataSource] = None,
    ) -> None:
        self._schema: Dict[str, Any] = dict(schema or {})
        self.config: Dict[str, Any] = dict(config or {})
        self.target_column = target_col or self._schema.get("target")
        self.data_source: Optional[DataSource] = data
        self.feature_names_ = []
        self.logger = logging.getLogger(f"evomind.adapter.{self.__class__.__name__}")
        self.baseline_model_ = None

    # ------------------------------------------------------------------ Schema
    @property
    def schema(self) -> Dict[str, Any]:
        """Return the profiler supplied schema metadata."""

        return self._schema

    @schema.setter
    def schema(self, value: Optional[Dict[str, Any]]) -> None:
        self._schema = dict(value or {})
        if not self.target_column:
            target = self._schema.get("target")
            if isinstance(target, str):
                self.target_column = target

    # ------------------------------------------------------------------ IO API
    def set_data_source(self, data: DataSource | None) -> None:
        """Update the datasource used by :meth:`load_data`."""

        self.data_source = data

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Load the raw dataset as a :class:`pandas.DataFrame`.

        Implementations may read from CSV/Parquet files, APIs, or in-memory
        frames.  The data source is exposed via :attr:`data_source`.
        """

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply domain specific cleaning/feature engineering.

        The returned dataframe **must** include the target column referenced by
        :attr:`target_column`. Feature columns should be numeric to ensure the
        evolution engine can consume them without additional coercion.
        """

    # --------------------------------------------------------------- Baselines
    @abstractmethod
    def train(self, X, y) -> Any:
        """Train a lightweight baseline model for quick benchmarking."""

    @abstractmethod
    def evaluate(self, model, X_test, y_test) -> Dict[str, float]:
        """Compute evaluation metrics for a trained model."""

    # -------------------------------------------------------------- AutoML API
    def build_model(self, genome: Genome, input_dim: int, output_dim: int) -> nn.Module:
        """
        Construct a PyTorch model from an EvoMind genome.

        Adapters can override this hook to inject domain aware modules (e.g.,
        recurrent layers for time series).  The default implementation builds a
        fully connected network suitable for tabular data.
        """

        layers: list[nn.Module] = []
        in_dim = input_dim
        for layer in genome.layers:
            if layer.layer_type == "dense":
                units = int(layer.params["units"])
                layers.append(nn.Linear(in_dim, units))
                activation = layer.params.get("activation", "relu").lower()
                layers.append(self._activation(activation))
                in_dim = units
            elif layer.layer_type == "dropout":
                layers.append(nn.Dropout(p=float(layer.params.get("p", 0.2))))
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def fitness(self, metrics: Dict[str, float]) -> float:
        """Derive a scalar fitness score from metric dictionaries."""

        loss = float(metrics.get("val_loss", metrics.get("rmse", 0.0)))
        return -loss

    # --------------------------------------------------------- Reporting hooks
    def summarize_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return dataset level statistics for insight generation."""

        target = self.target_column or (df.columns[-1] if not df.empty else "target")
        safe_df = df.copy()
        if target not in safe_df.columns:
            raise EvoMindAdapterError(
                "Adapter preprocess() must return a valid DataFrame with target_col present.",
                context={"target": target, "columns": list(safe_df.columns)},
            )
        return summarize_dataframe(safe_df, target=target)

    def generate_profile(self, df: pd.DataFrame, output_path: Path) -> Path:
        """Generate a lightweight profile report."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        return generate_profile_report(df, output_path)

    # ----------------------------------------------------------------- Helpers
    def prepare_training_frame(self) -> pd.DataFrame:
        """Convenience wrapper used by the orchestration layer."""

        df = self.load_data()
        processed = self.preprocess(df)
        if not isinstance(processed, pd.DataFrame):
            raise EvoMindAdapterError(
                "Adapter preprocess() must return a pandas.DataFrame.",
                context={"adapter": self.__class__.__name__},
            )
        if not self.target_column or self.target_column not in processed.columns:
            raise EvoMindAdapterError(
                "Adapter preprocess() must return a valid DataFrame with target_col present.",
                context={
                    "adapter": self.__class__.__name__,
                    "target_col": self.target_column,
                    "columns": list(processed.columns),
                },
            )
        return processed

    @staticmethod
    def _activation(name: str) -> nn.Module:
        mapping = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.2),
        }
        return mapping.get(name.lower(), nn.ReLU())


# Backwards compatibility alias used across the codebase.
BaseTaskAdapter = BaseAdapter

__all__ = ["BaseAdapter", "BaseTaskAdapter"]
