from typing import Any, Callable, Dict

import jax.numpy as jnp

from .data import MultiAgentTransitions

FeaturisingFn = Callable[[jnp.ndarray], jnp.ndarray]

WeightingFn = Callable[[MultiAgentTransitions, Dict[str, Any]], jnp.ndarray]
