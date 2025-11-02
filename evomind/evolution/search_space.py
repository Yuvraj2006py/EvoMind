"""
Search space definition for EvoMind genomes.

The search space enumerates candidate building blocks that can be mixed and
matched to create neural network architectures.  The initial implementation
focuses on fully connected and recurrent modules because they keep the demo
lightweight while still showcasing how the evolutionary process explores
different ideas.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import choice, randint
from typing import Any, Dict, List


@dataclass
class LayerSpec:
    """Describe a single layer within a genome."""

    layer_type: str
    params: Dict[str, Any]


class SearchSpace:
    """Utility for sampling layers and genomes from predefined pools."""

    def __init__(self) -> None:
        # A minimal yet expressive set of candidate layer templates.
        self.layer_pool: Dict[str, List[Dict[str, Any]]] = {
            "dense": [
                {"units": 64, "activation": "relu"},
                {"units": 128, "activation": "relu"},
                {"units": 256, "activation": "gelu"},
            ],
            "dropout": [
                {"p": 0.1},
                {"p": 0.2},
                {"p": 0.3},
            ],
        }

    def sample_layer(self, layer_type: str) -> LayerSpec:
        """Sample a random configuration of the requested layer type."""
        params = choice(self.layer_pool[layer_type])
        return LayerSpec(layer_type=layer_type, params=params)

    def random_layer(self) -> LayerSpec:
        """Draw a random layer from the entire pool."""
        layer_type = choice(list(self.layer_pool.keys()))
        return self.sample_layer(layer_type)

    def random_genome_spec(self, min_layers: int = 2, max_layers: int = 4) -> List[LayerSpec]:
        """Sample a list of layers that describe a candidate architecture."""
        depth = randint(min_layers, max_layers)
        return [self.random_layer() for _ in range(depth)]
