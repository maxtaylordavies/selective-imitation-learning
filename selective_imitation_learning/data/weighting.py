from functools import partial
from typing import Any, Callable, Dict

from jax import jit, vmap, Array
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm

from .transitions import SplitMultiAgentTransitions
from ..utils import cosine_sim, to_simplex
from ..types import FeaturisingFn


def weight_agents_uniform(omegas: Array, omegas_self: Array) -> Array:
    n_agents = omegas.shape[0]
    return jnp.ones(n_agents) / n_agents


def weight_agents_sim(omegas: Array, omegas_self: Array) -> Array:
    n_agents = omegas.shape[0]
    weights = jnp.zeros(n_agents)
    for i in range(n_agents):
        omega = to_simplex(omegas[i])
        weights = weights.at[i].set(cosine_sim(omega, omegas_self))
    return to_simplex(weights)


def transition_ll(f, obs, next_obs, a, omegas):
    obs, next_obs = obs.flatten(), next_obs.flatten()
    delta_f = f(next_obs) - f(obs)
    return jnp.dot(to_simplex(omegas[a]), delta_f)


@partial(jit, static_argnums=(0,))
def transitions_ll(
    f: FeaturisingFn,
    obss: Array,
    next_obss: Array,
    agents: Array,
    omegas: Array,
):
    return jnp.sum(
        vmap(transition_ll, in_axes=(None, 0, 0, 0, None))(
            f, obss, next_obss, agents, omegas
        )
    )


def dataset_ll(
    transitions: SplitMultiAgentTransitions, f: FeaturisingFn, omegas: Array
):
    return transitions_ll(
        f, transitions.obs, transitions.next_obs, transitions.agent_idxs, omegas
    )
