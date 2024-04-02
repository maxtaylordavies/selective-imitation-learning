from functools import partial
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import torch
from tqdm import tqdm

from .transitions import MultiAgentTransitions
from ..utils import cosine_sim, to_simplex
from ..types import FeaturisingFn


def weight_agents_uniform(
    transitions: MultiAgentTransitions, other_data: Dict[str, Any]
) -> jnp.ndarray:
    num_agents = len(jnp.unique(transitions.agent_idxs))
    return jnp.ones(num_agents) / num_agents


# def weight_agents_by_preference(
#     transitions: MultiAgentTransitions, other_data: Dict[str, Any]
# ) -> np.ndarray:
#     assert "own_preferences" in other_data
#     prefs = infer_preferences(jr.PRNGKey(0), transitions)
#     weights = np.array(
#         [cosine_sim(other_data["own_preferences"], prefs[a]) for a in prefs]
#     )
#     return weights / np.sum(weights)


# def infer_preferences_old(
#     rng_key, transitions: MultiAgentTransitions
# ) -> Dict[int, np.ndarray]:
#     def fruit_choice_model(data):
#         transitions, beta = data
#         num_agents = len(np.unique(transitions.agent_idxs))
#         fruit_counts = get_fruit_counts(transitions)
#         num_fruit_types = len(fruit_counts[0])

#         u = numpyro.sample(
#             "u",
#             dist.MultivariateNormal(
#                 jnp.zeros((num_agents, num_fruit_types)) + 0.5,
#                 1.0 * jnp.eye(num_fruit_types),
#             ),
#         )

#         p = jnp.exp(u / beta)
#         p /= jnp.sum(p, axis=-1, keepdims=True)

#         log_prob = 0
#         for a in range(num_agents):
#             log_prob += dist.Multinomial(
#                 probs=p[a], total_count=fruit_counts[a].sum()
#             ).log_prob(fruit_counts[a])
#         numpyro.factor(
#             "obs",
#             log_prob,
#         )

#     kernel = numpyro.infer.NUTS(fruit_choice_model)
#     mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=1000)
#     mcmc.run(rng_key, (transitions, 0.1))
#     samples = mcmc.get_samples()
#     u = samples["u"].mean(axis=0)

#     return {a: to_simplex(u[a]) for a in range(len(u))}


# def get_fruit_counts(transitions: MultiAgentTransitions) -> Dict[int, np.ndarray]:
#     def get_fruit_locs(obs):
#         rows, cols = np.nonzero((obs > 0) & (obs < 1))
#         return {obs[r, c]: (r, c) for r, c in zip(rows, cols)}

#     all_fruits, fruit_counts = set(), {}
#     for i in tqdm(range(len(transitions))):
#         if transitions.dones[i]:
#             continue

#         a = transitions.agent_idxs[i]
#         fruit_counts[a] = fruit_counts.get(a, {})

#         obs, next_obs = transitions.obs[i], transitions.next_obs[i]
#         locs, next_locs = get_fruit_locs(obs), get_fruit_locs(next_obs)
#         for fruit in locs:
#             if next_locs[fruit] != locs[fruit]:
#                 all_fruits.add(fruit)
#                 fruit_counts[a][fruit] = fruit_counts[a].get(fruit, 0) + 1

#     all_fruits = sorted(list(all_fruits))
#     for a in fruit_counts:
#         tmp = np.zeros_like(all_fruits)
#         for i, fruit in enumerate(all_fruits):
#             tmp[i] = fruit_counts[a].get(fruit, 0)
#         fruit_counts[a] = tmp

#     return fruit_counts


# def infer_agent_embeddings(
#     rng_key,
#     transitions: MultiAgentTransitions,
#     f: Callable[[np.ndarray], np.ndarray],
# ) -> Dict[int, np.ndarray]:
#     delta_f_cache = {}

#     def model(data):
#         transitions, f, n_agents, n_samples = data
#         n_features = f(transitions.obs[0]).shape[0]

#         ws = numpyro.sample(
#             "ws",
#             dist.MultivariateNormal(
#                 jnp.zeros((n_agents, n_features)) + 0.5,
#                 1.0 * jnp.eye(n_features),
#             ),
#         )

#         D = transitions if n_samples is None else transitions.sample_uniform(n_samples)
#         numpyro.factor("obs", dataset_log_likelihood(D, f, ws, delta_f_cache))

#     n_agents = len(jnp.unique(transitions.agent_idxs))

#     kernel = numpyro.infer.NUTS(model)
#     mcmc = numpyro.infer.MCMC(kernel, num_warmup=1000, num_samples=2000)
#     mcmc.run(rng_key, (transitions, f, n_agents, None))
#     samples = mcmc.get_samples()
#     ws = samples["ws"].mean(axis=0)

#     return {a: to_simplex(ws[a]) for a in range(len(ws))}


def transition_ll(f, obs, next_obs, a, omegas):
    delta_f = f(next_obs) - f(obs)
    return jnp.dot(to_simplex(omegas[a]), delta_f)


vmapped_transition_ll = jax.vmap(transition_ll, in_axes=(None, 0, 0, 0, None))


@partial(jax.jit, static_argnums=(0,))
def transitions_ll(
    f: FeaturisingFn,
    obss: jnp.ndarray,
    next_obss: jnp.ndarray,
    agents: jnp.ndarray,
    omegas: jnp.ndarray,
):
    return jnp.sum(vmapped_transition_ll(f, obss, next_obss, agents, omegas))


def dataset_ll(
    transitions: MultiAgentTransitions, f: FeaturisingFn, omegas: jnp.ndarray
):
    return transitions_ll(
        f, transitions.obs, transitions.next_obs, transitions.agent_idxs, omegas
    )
