import time

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from selective_imitation_learning.environments import FruitWorld, featurise
from selective_imitation_learning.data import MultiAgentTransitions
from selective_imitation_learning.utils import to_simplex, manhattan_dist
from selective_imitation_learning.networks import MLP

sns.set_theme(style="darkgrid")

fruit_prefs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
env_id = "FruitWorld-v0"
env_kwargs = {
    "grid_size": 7,
    "num_fruit": 3,
    "fruit_preferences": fruit_prefs[0],
    # "fruit_loc_means": np.array([[0, 0], [0, 6], [6, 3]]),
    "fruit_loc_means": np.array([[2, 2], [2, 4], [4, 3]]),
    "fruit_loc_stds": 2 * np.ones(3),
    "max_steps": 50,
    "render_mode": "human",
}


def featurise(obs: jnp.ndarray) -> jnp.ndarray:
    agent_pos = jnp.array(jnp.unravel_index(jnp.argmax(obs == -1.0), obs.shape))
    fruit_pos = jnp.array(
        [jnp.unravel_index(jnp.argmax(obs == i + 1), obs.shape) for i in range(3)]
    )
    feats = jnp.array([-manhattan_dist(agent_pos, f) for f in fruit_pos])
    return feats


def compute_action(obs, fruit_prefs):
    fruit = np.argmax(fruit_prefs) + 1
    own_pos = np.array(np.unravel_index(np.argmax(obs == -1.0), obs.shape))
    fruit_pos = np.array(np.unravel_index(np.argmax(obs == fruit), obs.shape))
    delta_x, delta_y = jnp.sign(fruit_pos - own_pos)

    poss_actions = []
    if delta_x == -1:
        poss_actions.append(0)
    if delta_x == 1:
        poss_actions.append(1)
    if delta_y == -1:
        poss_actions.append(2)
    if delta_y == 1:
        poss_actions.append(3)

    if len(poss_actions) == 0:
        poss_actions = [0, 1, 2, 3]

    return np.random.choice(poss_actions)


def generate_expert_data(env, min_ts_per_agent):
    obss, acts, next_obss, dones, infos, a_idxs = [], [], [], [], [], []
    pbar = tqdm(total=3 * min_ts_per_agent)
    for a in range(3):
        ts_done = 0
        while ts_done < min_ts_per_agent:
            obs, _ = env.reset()
            done = False
            while not done:
                obss.append(obs)

                act = compute_action(obs, fruit_prefs[a])
                next_obs, _, done, _, info = env.step(act)

                acts.append(act)
                next_obss.append(next_obs)
                dones.append(done)
                infos.append(info)
                a_idxs.append(a)

                obs = next_obs
                ts_done += 1
                pbar.update(1)

    transitions = MultiAgentTransitions(
        obs=jnp.array(obss),
        acts=jnp.array(acts),
        next_obs=jnp.array(next_obss),
        dones=jnp.array(dones),
        infos=np.array(infos),
        agent_idxs=jnp.array(a_idxs),
    )
    print(f"generated {len(transitions)} transitions")
    return transitions


@jax.jit
def transition_ll(delta_f, omegas, a):
    # return to_simplex(delta_f) @ to_simplex(omegas[a])
    ll = 2 * (to_simplex(delta_f) @ to_simplex(omegas[a]))
    for i in range(3):
        ll -= to_simplex(delta_f) @ to_simplex(omegas[i])
    return ll


@eqx.filter_jit
def loss_fn(f, omegas, obss, next_obss, a_idxs):
    fs = jax.vmap(f)(obss)
    f_nexts = jax.vmap(f)(next_obss)
    delta_fs = f_nexts - fs
    lls = jax.vmap(transition_ll, in_axes=(0, None, 0))(delta_fs, omegas, a_idxs)
    return -jnp.sum(lls)


def sample_episodes(key, transitions, n, ep_len=50):
    ep_start_idxs = jnp.where(transitions.dones)[0] + 1
    ep_start_idxs = jnp.concatenate([jnp.array([0]), ep_start_idxs[:-1]])
    starts = jax.random.choice(key, ep_start_idxs, shape=(n,), replace=False)

    eps = []
    for start in starts:
        eps.append(transitions[start : start + ep_len])
    return eps


def compute_agent_probs(f, omegas, episode):
    assert len(jnp.unique(episode.agent_idxs)) == 1
    delta_f_sum = jnp.zeros(3)
    agent_probs = [0.0, 0.0, 0.0]
    for i in range(len(episode)):
        obs, next_obs = episode.obs[i], episode.next_obs[i]
        delta_f = to_simplex(f(next_obs) - f(obs))
        delta_f_sum += delta_f
        for a in range(3):
            agent_probs[a] += float(delta_f @ to_simplex(omegas[a]))
    return to_simplex(jnp.exp(jnp.array(agent_probs))), delta_f_sum


def do_eval(key, transitions, featuriser, omegas, n=100, print_fs=False):
    episodes, score, ref_score = sample_episodes(key, transitions, n), 0, 0
    for ep in episodes:
        assert len(jnp.unique(ep.agent_idxs)) == 1
        agent_probs, _ = compute_agent_probs(featuriser, omegas, ep)
        # ref_agent_probs, _ = compute_agent_probs(featurise, omegas, ep)
        if jnp.argmax(jnp.array(agent_probs)) == ep.agent_idxs[0]:
            score += 1
        # if jnp.argmax(jnp.array(ref_agent_probs)) == ep.agent_idxs[0]:
        #     ref_score += 1
    return score / n


def train(
    key,
    featuriser,
    omegas,
    train_data,
    test_data,
    n_iter=int(1e4),
    batch_size=int(1e4),
    log_interval=100,
    eval_interval=1000,
):
    f_optim = optax.adam(1e-2)
    f_state = f_optim.init(eqx.filter(featuriser, eqx.is_array))

    @eqx.filter_jit
    def do_train_step(featuriser, omegas, f_state, obss, next_obss, a_idxs):
        loss, f_grad = eqx.filter_value_and_grad(loss_fn)(
            featuriser, omegas, obss, next_obss, a_idxs
        )
        f_updates, f_state = f_optim.update(f_grad, f_state)
        featuriser = eqx.apply_updates(featuriser, f_updates)
        return featuriser, f_state, loss

    losses, evals = [], []
    keys = jr.split(key, n_iter)
    for i in tqdm(range(n_iter)):
        batch_idxs = jr.randint(keys[i], (batch_size,), 0, len(train_data))
        batch = train_data[batch_idxs]

        featuriser, f_state, loss = do_train_step(
            featuriser, omegas, f_state, batch.obs, batch.next_obs, batch.agent_idxs
        )
        losses.append(float(loss))

        if i % log_interval == 0:
            tqdm.write(f"loss: {loss}")
        if i % eval_interval == 0:
            eval_score = do_eval(keys[i], test_data, featuriser, omegas, n=50)
            tqdm.write(f"eval: {eval_score}")
            evals.append(eval_score)

    return featuriser, losses, evals


def compare_featurisers(f, f_target, transitions, n_samples=1000):
    data = {
        "learned_f_0": [],
        "learned_f_1": [],
        "learned_f_2": [],
        "target_f_0": [],
        "target_f_1": [],
        "target_f_2": [],
    }
    idxs = jr.choice(jr.PRNGKey(0), len(transitions), (n_samples,), replace=False)
    for i, idx in enumerate(idxs):
        obs, next_obs = test_data.obs[idx], test_data.next_obs[idx]
        learned_f = to_simplex(f(next_obs) - f(obs))
        target_f = to_simplex(f_target(next_obs) - f_target(obs))
        for j in range(3):
            data[f"learned_f_{j}"].append(float(learned_f[j]))
            data[f"target_f_{j}"].append(float(target_f[j]))
    data = pd.DataFrame(data)

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        for j in range(3):
            sns.regplot(data=data, x=f"target_f_{i}", y=f"learned_f_{j}", ax=axs[i, j])
            axs[i, j].set_xlabel(f"target_f_{i}")
            axs[i, j].set_ylabel(f"learned_f_{j}")
    fig.tight_layout()
    plt.show()


def compare_featurisers_2(f, f_target, transitions, n_samples=100):
    data = {
        "learned_f_0": [],
        "learned_f_1": [],
        "learned_f_2": [],
        "target_f_0": [],
        "target_f_1": [],
        "target_f_2": [],
    }

    eps = sample_episodes(jr.PRNGKey(0), transitions, n_samples)
    for ep in tqdm(eps):
        _, learned_f = compute_agent_probs(f, omegas, ep)
        _, target_f = compute_agent_probs(f_target, omegas, ep)
        for j in range(3):
            data[f"learned_f_{j}"].append(float(learned_f[j]))
            data[f"target_f_{j}"].append(float(target_f[j]))
    data = pd.DataFrame(data)

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        for j in range(3):
            sns.regplot(data=data, x=f"target_f_{i}", y=f"learned_f_{j}", ax=axs[i, j])
            axs[i, j].set_xlabel(f"target_f_{i}")
            axs[i, j].set_ylabel(f"learned_f_{j}")
    fig.tight_layout()
    plt.show()


# set key for jax.random
seed = int(time.time())
key = jr.PRNGKey(seed)

# sample expert transitions
env = gym.make(env_id, **env_kwargs)
train_data = generate_expert_data(env, int(1e5))
test_data = generate_expert_data(env, int(1e4))

# set omegas
omegas = jnp.array(fruit_prefs)

# find network initialisation with lowest starting loss
best_init_loss, best_featuriser, best_key = jnp.inf, None, None
for k in tqdm(jr.split(key, 50)):
    featuriser = MLP(
        k,
        in_size=49,
        hidden_size=32,
        out_size=3,
        num_hidden_layers=1,
    )
    loss = loss_fn(
        featuriser, omegas, train_data.obs, train_data.next_obs, train_data.agent_idxs
    )
    if loss < best_init_loss:
        best_init_loss = loss
        best_featuriser = featuriser
        best_key = k
print(f"best init loss: {best_init_loss}")
featuriser, key = best_featuriser, best_key

# visualise featuriser before training
compare_featurisers_2(featuriser, featurise, test_data)

# train network
featuriser, losses, evals = train(
    key,
    featuriser,
    omegas,
    train_data,
    test_data,
    n_iter=int(1e4),
    batch_size=int(1e4),
)

# plot training curves
sns.lineplot(x=range(len(losses)), y=losses)
plt.show()
sns.lineplot(x=range(len(evals)), y=evals)
plt.show()

# visualise featuriser after training
compare_featurisers_2(featuriser, featurise, test_data)


# episodes = sample_episodes(key, test_data, 20)
# for ep in episodes:
#     print(f"agent: {ep.agent_idxs[0]}")
#     pred_agent_probs, pred_f = compute_agent_probs(featuriser, omegas, ep)
#     true_agent_probs, true_f = compute_agent_probs(featurise, omegas, ep)
#     fig, axs = plt.subplots(2, 2, figsize=(8, 8))
#     sns.barplot(x=["f_0", "f_1", "f_2"], y=to_simplex(pred_f), ax=axs[0, 0])
#     sns.barplot(x=["f_0", "f_1", "f_2"], y=to_simplex(true_f), ax=axs[0, 1])
#     sns.barplot(x=["a_0", "a_1", "a_2"], y=pred_agent_probs, ax=axs[1, 0])
#     sns.barplot(x=["a_0", "a_1", "a_2"], y=true_agent_probs, ax=axs[1, 1])
#     axs[0, 0].set_title("Learned features")
#     axs[0, 1].set_title("Target features")
#     axs[1, 0].set_title("Predicted agent probs")
#     axs[1, 1].set_title("Target agent probs")
#     plt.show()

# for a in range(3):
#     obs, _ = env.reset()
#     prev_obs = obs.copy()
#     for i in range(50):
#         act = compute_action(prev_obs, fruit_prefs[a])
#         obs, _, _, _, _ = env.step(act)

#         env.render()

#         f_ref = featurise(obs) - featurise(prev_obs)
#         f_pred = featuriser(obs) - featuriser(prev_obs)

#         fig, axs = plt.subplots(2, 1, figsize=(4, 8), sharex=True)
#         sns.barplot(x=["f_0", "f_1", "f_2"], y=f_ref, ax=axs[0])
#         sns.barplot(x=["f_0", "f_1", "f_2"], y=f_pred, ax=axs[1])
#         plt.show()

#         prev_obs = obs.copy()
