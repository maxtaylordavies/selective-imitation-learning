from typing import Any, Callable, Dict

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
from tqdm import tqdm

from .transitions import MultiAgentTransitions
from ..utils import cosine_sim, to_simplex

WeightingFn = Callable[[MultiAgentTransitions, Dict[str, Any]], np.ndarray]


def weight_agents_uniform(
    transitions: MultiAgentTransitions, other_data: Dict[str, Any]
) -> np.ndarray:
    num_agents = len(np.unique(transitions.agent_idxs))
    return np.ones(num_agents) / num_agents


def weight_agents_by_preference(
    transitions: MultiAgentTransitions, other_data: Dict[str, Any]
) -> np.ndarray:
    assert "own_preferences" in other_data
    prefs = infer_preferences(jr.PRNGKey(0), transitions)
    weights = np.array(
        [cosine_sim(other_data["own_preferences"], prefs[a]) for a in prefs]
    )
    return weights / np.sum(weights)


def infer_preferences(
    rng_key, transitions: MultiAgentTransitions
) -> Dict[int, np.ndarray]:
    kernel = numpyro.infer.NUTS(fruit_choice_model)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=1000)
    mcmc.run(rng_key, (transitions, 0.1))
    samples = mcmc.get_samples()
    u = samples["u"].mean(axis=0)

    return {a: to_simplex(u[a]) for a in range(len(u))}


def fruit_choice_model(data):
    transitions, beta = data
    num_agents = len(np.unique(transitions.agent_idxs))
    fruit_counts = get_fruit_counts(transitions)
    num_fruit_types = len(fruit_counts[0])

    u = numpyro.sample(
        "u",
        dist.MultivariateNormal(
            jnp.zeros((num_agents, num_fruit_types)) + 0.5,
            1.0 * jnp.eye(num_fruit_types),
        ),
    )

    p = jnp.exp(u / beta)
    p /= jnp.sum(p, axis=-1, keepdims=True)

    log_prob = 0
    for a in range(num_agents):
        log_prob += dist.Multinomial(
            probs=p[a], total_count=fruit_counts[a].sum()
        ).log_prob(fruit_counts[a])
    numpyro.factor(
        "obs",
        log_prob,
    )


def get_fruit_counts(transitions: MultiAgentTransitions) -> Dict[int, np.ndarray]:
    def get_fruit_locs(obs):
        rows, cols = np.nonzero((obs > 0) & (obs < 1))
        return {obs[r, c]: (r, c) for r, c in zip(rows, cols)}

    all_fruits, fruit_counts = set(), {}
    for i in tqdm(range(len(transitions))):
        if transitions.dones[i]:
            continue

        a = transitions.agent_idxs[i]
        fruit_counts[a] = fruit_counts.get(a, {})

        obs, next_obs = transitions.obs[i], transitions.next_obs[i]
        locs, next_locs = get_fruit_locs(obs), get_fruit_locs(next_obs)
        for fruit in locs:
            if next_locs[fruit] != locs[fruit]:
                all_fruits.add(fruit)
                fruit_counts[a][fruit] = fruit_counts[a].get(fruit, 0) + 1

    all_fruits = sorted(list(all_fruits))
    for a in fruit_counts:
        tmp = np.zeros_like(all_fruits)
        for i, fruit in enumerate(all_fruits):
            tmp[i] = fruit_counts[a].get(fruit, 0)
        fruit_counts[a] = tmp

    return fruit_counts


def infer_preferences_hacky(
    transitions: MultiAgentTransitions,
) -> Dict[int, np.ndarray]:
    fruit_counts = get_fruit_counts(transitions)
    return {a: fruit_counts[a] / np.sum(fruit_counts[a]) for a in fruit_counts}
