"""Tests covering the EvoMind configuration loader behaviour."""

from pathlib import Path

import pytest

from evomind.utils import ConfigLoader


def test_config_loader_merges_task_overrides() -> None:
    """Task specific configuration should override global defaults."""
    loader = ConfigLoader(Path("configs/global.yaml"))
    config = loader.load(Path("configs/retail.yaml")).to_dict()
    assert config["trainer"]["epochs"] == 4
    assert config["population_size"] == 20


def test_config_loader_accepts_dict_overrides() -> None:
    loader = ConfigLoader({"engine": {"generations": 2}})
    config = loader.load(overrides={"engine": {"population": 5}}).to_dict()
    assert config["engine"]["generations"] == 2
    assert config["engine"]["population"] == 5


def test_config_loader_unknown_section_raises() -> None:
    loader = ConfigLoader()
    with pytest.raises(ValueError) as err:
        loader.load(overrides={"unknown_section": {"foo": 1}})
    assert "unknown_section" in str(err.value)


def test_config_loader_unknown_key_raises() -> None:
    loader = ConfigLoader()
    with pytest.raises(ValueError) as err:
        loader.load(overrides={"engine": {"invalid_key": 1}})
    assert "engine.invalid_key" in str(err.value)
