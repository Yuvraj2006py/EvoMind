"""Top-level package exposing EvoMind SDK entrypoints."""

import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="visions.utils.monkeypatches.imghdr_patch",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="matplotlib.cbook",
)

from .pipelines import EvoMind, EvoMindResult

__all__ = ["EvoMind", "EvoMindResult"]
