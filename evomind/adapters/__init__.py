"""Adapter exports and registry helpers."""

from __future__ import annotations

from typing import Dict, Type

from .base_adapter import BaseAdapter, BaseTaskAdapter
from .registry import adapter_registry, get_adapter, list_adapters, register_adapter

TASK_REGISTRY: Dict[str, Type[BaseAdapter]] = {}


def register_task(name: str):
    """Backward compatible decorator mirroring :func:`register_adapter`."""

    def decorator(cls: Type[BaseAdapter]) -> Type[BaseAdapter]:
        register_adapter(name)(cls)
        TASK_REGISTRY[name] = cls
        return cls

    return decorator


__all__ = [
    "BaseAdapter",
    "BaseTaskAdapter",
    "adapter_registry",
    "register_adapter",
    "register_task",
    "get_adapter",
    "list_adapters",
    "TASK_REGISTRY",
]

# Import default adapters so they register themselves.
_IMPORTS = [
    "retail_forecasting",
    "marketing_adapter",
    "sports_adapter",
    "finance_adapter",
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
