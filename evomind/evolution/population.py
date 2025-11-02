"""
Population management utilities for EvoMind evolution cycles.

The `PopulationManager` is responsible for seeding initial genomes, storing
evaluation metadata, and providing helper methods that make selection and
sorting straightforward for the evolution engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence

from .genome import Genome
from .search_space import SearchSpace


@dataclass
class PopulationManager:
    """Container around a list of genomes with helper utilities."""

    search_space: SearchSpace
    population_size: int
    genomes: List[Genome] = field(default_factory=list)

    def seed(self) -> None:
        """Populate the manager with fresh random genomes."""
        self.genomes = [
            Genome(layers=self.search_space.random_genome_spec(), generation=0, parent_ids=[])
            for _ in range(self.population_size)
        ]

    def sort(self, key: Callable[[Genome], float], reverse: bool = True) -> None:
        """Sort genomes in place according to the provided key function."""
        self.genomes.sort(key=key, reverse=reverse)

    def top_k(self, k: int) -> Sequence[Genome]:
        """Return the best performing genomes."""
        return self.genomes[:k]

    def extend(self, offspring: Iterable[Genome]) -> None:
        """Add new genomes to the managed population."""
        self.genomes.extend(offspring)

    def trim(self) -> None:
        """Keep only the strongest genomes according to the configured population size."""
        self.genomes = self.genomes[: self.population_size]
