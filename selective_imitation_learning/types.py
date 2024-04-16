from typing import Any, Callable, Dict

import jax

from .data import MultiAgentTransitions

FeaturisingFn = Callable[[jax.Array], jax.Array]

# WeightingFn = Callable[[MultiAgentTransitions, Dict[str, Any]], jax.Array]

WeightingFn = Callable[[jax.Array, jax.Array], jax.Array]
