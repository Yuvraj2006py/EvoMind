"""
Tests for population management utilities.
"""

from evomind.evolution.population import PopulationManager
from evomind.evolution.search_space import SearchSpace


def test_population_seed_initialises_the_requested_size() -> None:
    """Seeding a population should create the requested number of genomes."""
    manager = PopulationManager(search_space=SearchSpace(), population_size=5)
    manager.seed()
    assert len(manager.genomes) == 5
