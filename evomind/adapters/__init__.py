"""
Task adapter registry for EvoMind.

This module exposes a decorator that automatically registers adapters so the
core framework can create them dynamically based on CLI or configuration input.
"""

from __future__ import annotations

from typing import Dict, Type

from .base_adapter import BaseTaskAdapter

TASK_REGISTRY: Dict[str, Type[BaseTaskAdapter]] = {}


def register_task(name: str):
    """Decorator used to register task adapters by name."""

    def decorator(cls: Type[BaseTaskAdapter]) -> Type[BaseTaskAdapter]:
        TASK_REGISTRY[name] = cls
        return cls

    return decorator


__all__ = ["BaseTaskAdapter", "TASK_REGISTRY", "register_task"]

# Import default adapters so they register themselves.
_IMPORTS = [
    "retail_forecasting",
    "classification_adapter",
    "regression_adapter",
    "vision_adapter",
    "nlp_sentiment_adapter",
    "multimodal_adapter",
]

for _module in _IMPORTS:
    try:
        __import__(f"{__name__}.{_module}")
    except ImportError:  # pragma: no cover - optional dependencies may be missing.
        continue

