"""Adapter registry used by EvoMind to locate domain plugins."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Type

from .base_adapter import BaseAdapter


class AdapterRegistry:
    """Light-weight registry storing adapter classes by name."""

    def __init__(self) -> None:
        self._registry: Dict[str, Type[BaseAdapter]] = {}

    def register(self, name: str, adapter_cls: Type[BaseAdapter]) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("Adapter name cannot be empty.")
        self._registry[key] = adapter_cls

    def get(self, name: str) -> Type[BaseAdapter]:
        key = name.strip().lower()
        if key not in self._registry:
            raise KeyError(
                f"Adapter '{name}' is not registered. Available adapters: {sorted(self._registry)}"
            )
        return self._registry[key]

    def available(self) -> Dict[str, Type[BaseAdapter]]:
        return dict(self._registry)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return name.strip().lower() in self._registry

    def __iter__(self) -> Iterable[str]:
        return iter(sorted(self._registry))


adapter_registry = AdapterRegistry()


def register_adapter(name: str):
    """Decorator registering adapters in the global registry."""

    def decorator(cls: Type[BaseAdapter]) -> Type[BaseAdapter]:
        adapter_registry.register(name, cls)
        return cls

    return decorator


def list_adapters() -> Dict[str, Type[BaseAdapter]]:
    return adapter_registry.available()


def get_adapter(name: str) -> Type[BaseAdapter]:
    return adapter_registry.get(name)


__all__ = ["adapter_registry", "AdapterRegistry", "register_adapter", "list_adapters", "get_adapter"]
