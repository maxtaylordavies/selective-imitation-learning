from typing import Optional

import jax
import jax.random as jr
import jax.numpy as jnp

from .transitions import MultiAgentTransitions


class BCBatchLoader:
    def __init__(
        self,
        transitions: MultiAgentTransitions,
        batch_size: int,
    ):
        self.transitions = transitions
        self.batch_size = batch_size
        self.sample_weights = jnp.ones(len(transitions)) / len(transitions)

    def set_sample_weights(self, agent_weights: jax.Array):
        ws = agent_weights[self.transitions.agent_idxs]
        _sum = ws.sum()
        if _sum == 0:
            self.sample_weights = jnp.ones(len(ws)) / len(ws)
        else:
            self.sample_weights = ws / _sum

    def sample_batch(
        self, agent_weights: Optional[jax.Array], uniform=False, size=None
    ) -> MultiAgentTransitions:
        if not uniform and agent_weights is not None:
            self.set_sample_weights(agent_weights)

        size = size or self.batch_size
        p = None if uniform else self.sample_weights
        return self._sample_batch(size, p)

    def _sample_batch(self, n, p) -> MultiAgentTransitions:
        tmp = jnp.arange(len(self.transitions))
        idxs = jr.choice(jr.PRNGKey(0), tmp, (n,), p=p)
        return self.transitions[idxs]
