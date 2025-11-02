"""
Representation of EvoMind genomes.

Genomes describe neural architectures using a high level list of `LayerSpec`
instructions.  They also carry bookkeeping metadata such as evaluation metrics
and fitness scores gathered during evolution.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from torch import nn

from .search_space import LayerSpec, SearchSpace

_genome_id = itertools.count()


@dataclass
class Genome:
    """Container representing an architecture candidate."""

    layers: List[LayerSpec]
    metrics: Dict[str, float] = field(default_factory=dict)
    fitness: Optional[float] = None
    trained_model: Optional[nn.Module] = field(default=None, repr=False)
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    id: int = field(default_factory=lambda: next(_genome_id))

    def clone(self) -> "Genome":
        """Create a shallow copy used during evolution operations."""
        return Genome(
            layers=list(self.layers),
            generation=self.generation,
            parent_ids=[self.id],
        )

    def mutate(self, search_space: SearchSpace, mutation_rate: float = 0.3) -> "Genome":
        """
        Apply simple random mutations to the genome.

        Mutation picks a random layer and replaces it with another configuration
        from the same layer pool.  The mutation rate controls the chance of an
        additional layer being inserted to marginally increase depth diversity.
        """

        mutated_layers = list(self.layers)
        if not mutated_layers:
            mutated_layers = search_space.random_genome_spec()
        else:
            from random import randrange

            position = randrange(len(mutated_layers))
            mutated_layers[position] = search_space.random_layer()
        # With a small probability append a fresh layer.
        from random import random

        if len(mutated_layers) < 6 and random() < mutation_rate:
            mutated_layers.append(search_space.random_layer())
        parents = list(self.parent_ids) if self.parent_ids else [self.id]
        return Genome(layers=mutated_layers, parent_ids=parents, generation=self.generation)

    @staticmethod
    def crossover(parent_a: "Genome", parent_b: "Genome") -> "Genome":
        """
        Perform single point crossover between two parent genomes.

        The implementation is intentionally lightweight yet showcases how
        recombination can produce new layer combinations from different parents.
        """

        split = max(1, min(len(parent_a.layers), len(parent_b.layers)) // 2)
        child_layers = parent_a.layers[:split] + parent_b.layers[split:]
        return Genome(
            layers=child_layers,
            parent_ids=[parent_a.id, parent_b.id],
            generation=max(parent_a.generation, parent_b.generation),
        )