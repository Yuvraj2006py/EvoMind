"""
Predefined configuration profiles for EvoMind.

Profiles provide convenient shortcuts for common experimentation modes
such as quick smoke tests or exhaustive searches. They are merged on top
of global defaults before user overrides are applied.
"""

from __future__ import annotations

from typing import Dict

from omegaconf import OmegaConf


PROFILES: Dict[str, Dict[str, object]] = {
    "fast": {
        "engine": {
            "generations": 1,
            "population": 6,
            "epochs": 1,
            "batch_size": 32,
            "parallel": False,
        },
        "insights": {
            "shap_sample_size": 50,
        },
        "reporting": {
            "export_formats": ["html"],
        },
    },
    "balanced": {
        "engine": {
            "generations": 6,
            "population": 20,
            "epochs": 3,
            "batch_size": 32,
            "parallel": True,
            "ensemble_top_k": 3,
        },
        "insights": {
            "shap_sample_size": 200,
        },
    },
    "exhaustive": {
        "engine": {
            "generations": 12,
            "population": 48,
            "epochs": 6,
            "batch_size": 32,
            "parallel": True,
            "bayes_rounds": 6,
            "ensemble_top_k": 5,
        },
        "insights": {
            "shap_sample_size": 400,
        },
        "reporting": {
            "export_formats": ["html", "pdf"],
        },
    },
}


def list_profiles() -> Dict[str, Dict[str, object]]:
    """Return a copy of the registered profiles."""

    return {name: OmegaConf.to_container(OmegaConf.create(conf), resolve=True) for name, conf in PROFILES.items()}


def get_profile(name: str) -> Dict[str, object]:
    """Return a profile configuration by name."""

    if name not in PROFILES:
        raise KeyError(f"Unknown profile '{name}'. Available profiles: {list(PROFILES)}")
    return OmegaConf.to_container(OmegaConf.create(PROFILES[name]), resolve=True)
