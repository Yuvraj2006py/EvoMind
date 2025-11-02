"""
Tests covering the EvoMind configuration loader behaviour.
"""

from pathlib import Path

from evomind.utils import ConfigLoader


def test_config_loader_merges_task_overrides() -> None:
    """Task specific configuration should override global defaults."""
    loader = ConfigLoader(Path("configs/global.yaml"))
    config = loader.load(Path("configs/retail.yaml")).to_dict()
    assert config["trainer"]["epochs"] == 4
    assert config["population_size"] == 20
