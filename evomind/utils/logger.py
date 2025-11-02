"""
Unified logging utilities that wrap Loguru and MLflow.

The `ExperimentLogger` offers a small convenience layer that the rest of the
framework can use without worrying about tracking URIs or missing optional
dependencies.  Metrics and parameters are forwarded to MLflow when available,
while Loguru handles rich console output.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional

from loguru import logger

try:
    import mlflow
except ImportError:  # pragma: no cover - fallback path is best effort only.
    mlflow = None  # type: ignore[assignment]
    logger.warning("MLflow is not installed. Tracking will be disabled.")


class ExperimentLogger:
    """Thin convenience wrapper around Loguru and MLflow."""

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

    def _ensure_mlflow(self) -> None:
        """Configure the MLflow tracking URI and experiment if MLflow is available."""
        if mlflow is None:
            return
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    @contextmanager
    def start_run(self, run_name: str, params: Optional[Dict[str, str]] = None) -> Iterator[None]:
        """
        Context manager that opens and closes an MLflow run while emitting log messages.

        When MLflow is not available the context still works, so upstream code can rely
        on the same interface without extra guards.
        """

        logger.info("Starting EvoMind run: {}", run_name)
        if mlflow:
            self._ensure_mlflow()
            with mlflow.start_run(run_name=run_name):
                if params:
                    mlflow.log_params(params)
                yield
        else:  # pragma: no cover - executed only when MLflow is missing.
            yield
        logger.info("Completed EvoMind run: {}", run_name)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Emit metrics to both the console and MLflow if available."""
        logger.debug("Metrics@{}: {}", step if step is not None else "-", metrics)
        if mlflow:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        """Record an artifact with MLflow when available."""
        if mlflow and path.exists():
            mlflow.log_artifact(str(path))

    def log_message(self, message: str) -> None:
        """Log a simple info message."""
        logger.info(message)
