from typing import Any, Callable, Dict

import jax

from .data import SplitMultiAgentTransitions

FeaturisingFn = Callable[[jax.Array], jax.Array]

# WeightingFn = Callable[[SplitMultiAgentTransitions, Dict[str, Any]], jax.Array]

WeightingFn = Callable[[jax.Array, jax.Array], jax.Array]
