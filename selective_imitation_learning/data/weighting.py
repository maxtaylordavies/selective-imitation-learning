from functools import partial
from typing import Any, Callable, Dict

from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm

from .transitions import MultiAgentTransitions
from ..utils import cosine_sim, to_simplex
from ..types import FeaturisingFn


def weight_agents_uniform(
    transitions: MultiAgentTransitions, other_data: Dict[str, Any]
) -> jnp.ndarray:
    num_agents = len(jnp.unique(transitions.agent_idxs))
    return jnp.ones(num_agents) / num_agents


def transition_ll(f, obs, next_obs, a, omegas):
    delta_f = f(next_obs) - f(obs)
    return jnp.dot(to_simplex(omegas[a]), delta_f)


# vmapped_transition_ll = vmap(transition_ll, in_axes=(None, 0, 0, 0, None))


@partial(jit, static_argnums=(0,))
def transitions_ll(
    f: FeaturisingFn,
    obss: jnp.ndarray,
    next_obss: jnp.ndarray,
    agents: jnp.ndarray,
    omegas: jnp.ndarray,
):
    return jnp.sum(
        vmap(transition_ll, in_axes=(None, 0, 0, 0, None))(
            f, obss, next_obss, agents, omegas
        )
    )


def dataset_ll(
    transitions: MultiAgentTransitions, f: FeaturisingFn, omegas: jnp.ndarray
):
    return transitions_ll(
        f, transitions.obs, transitions.next_obs, transitions.agent_idxs, omegas
    )
