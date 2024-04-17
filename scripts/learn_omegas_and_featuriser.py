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
from PIL import Image


from selective_imitation_learning.environments import FruitWorld, featurise
from selective_imitation_learning.environments.fruitworld import expert_policy
from selective_imitation_learning.data import (
    SplitMultiAgentTransitions,
    generate_split_dataset,
)
from selective_imitation_learning.utils import (
    is_power,
    to_range,
    to_simplex,
    manhattan_dist,
    fig_to_pil_image,
    images_to_gif,
)
from selective_imitation_learning.networks import MLP
from selective_imitation_learning.constants import ENV_CONSTANTS

sns.set_theme(style="darkgrid")

fruit_prefs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
env_id = "FruitWorld-v0"
env_kwargs = {
    "grid_size": 7,
    "num_fruit": 3,
    "fruit_prefs": fruit_prefs[0],
    # "fruit_loc_means": np.array([[0, 0], [0, 6], [6, 3]]),
    "fruit_loc_means": np.array([[3, 3], [3, 3], [3, 3]]),
    "fruit_loc_stds": 1 * np.ones(3),
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


@jax.jit
def transition_ll(delta_f, omegas, a):
    # higher is better
    delta_f = to_simplex(delta_f)
    ll = delta_f @ omegas[a]
    tmp = jnp.dot(omegas, omegas.T)
    o_penalty = tmp.sum() - jnp.trace(tmp)
    return ll - (0.1 * o_penalty)
    # ll = 2 * (to_simplex(delta_f) @ to_simplex(omegas[a]))
    # for i in range(3):
    #     ll -= to_simplex(delta_f) @ to_simplex(omegas[i])
    # return ll


@eqx.filter_jit
def loss_fn(f, omegas, obss, next_obss, a_idxs):
    fs = jax.vmap(f)(obss)
    f_nexts = jax.vmap(f)(next_obss)
    delta_fs = f_nexts - fs
    return -jnp.sum(
        jax.vmap(transition_ll, in_axes=(0, None, 0))(
            delta_fs, jnp.array([to_simplex(omegas[a]) for a in range(3)]), a_idxs
        )
    )


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
    episodes, score = sample_episodes(key, transitions, n), 0
    for ep in episodes:
        assert len(jnp.unique(ep.agent_idxs)) == 1
        agent_probs, _ = compute_agent_probs(featuriser, omegas, ep)
        if jnp.argmax(jnp.array(agent_probs)) == ep.agent_idxs[0]:
            score += 1
    return score / n


def featuriser_histogram(f, omegas, episodes, bins=125):
    # make a 3d array to store histogram counts
    # array should be of shape (bins**(1/3), bins**(1/3), bins**(1/3))
    # where the 3 dimensions correspond to the 3 features
    side = round(bins ** (1 / 3))
    hist, fss = np.zeros((side, side, side)), []

    for ep in episodes:
        _, fs = compute_agent_probs(f, omegas, ep)
        fs = fs / np.linalg.norm(fs)
        fss.append(fs * side)
        idxs = np.clip(np.floor(fs * side).astype(np.int32), 0, side - 1)
        hist[idxs[0], idxs[1], idxs[2]] += 1

    return hist, fss


def visualise_training_progress(
    f, omegas, transitions, loss_data, eval_data, n_samples=300, bins=1000, title=""
):
    assert is_power(bins, 3), "number of bins must be integer cube"

    # create figure
    fig = plt.figure(figsize=(10, 10))
    axs = [
        fig.add_subplot(221, projection="3d"),  # featuriser
        fig.add_subplot(222, projection="3d"),  # omegas
        fig.add_subplot(223),  # loss
        fig.add_subplot(224),  # eval
    ]

    # compute feature historgrams
    side = round(bins ** (1 / 3))
    n_agents = len(np.unique(transitions.agent_idxs))
    hists, pointss = np.zeros((side, side, side, n_agents)), []
    for a_idx in range(n_agents):
        tmp = transitions[transitions.agent_idxs == a_idx]
        eps = sample_episodes(jr.PRNGKey(0), tmp, n_samples // 3)
        hist, points = featuriser_histogram(f, omegas, eps, bins=bins)
        hists[..., a_idx] = hist
        pointss.append(points)

    # now transform hists into an array of rgb values
    # each bin should be mapped to a colour, which is a
    # weighted sum of the 3 fruit colours (weighted by the bin value per agent)
    colour_basis = np.array(ENV_CONSTANTS["fruit_colours"]) / 255
    rgb = np.zeros((side, side, side, 3))
    for a_idx in range(n_agents):
        rgb += hists[..., a_idx, None] * colour_basis[a_idx]

    # normalise colours
    hists_sum = np.sum(hists, axis=-1)
    tmp = hists_sum.copy()
    tmp[tmp == 0] = 1
    rgb /= tmp[..., None]

    # use the sum of the histograms to set the alpha channel
    alphas = hists_sum / np.max(hists_sum)
    rgba = np.concatenate([rgb, alphas[..., None]], axis=-1)

    # plot
    axs[0].voxels(alphas > 0, facecolors=rgba, edgecolors=rgba)
    for a_idx, points in enumerate(pointss):
        for p in points:
            axs[0].scatter(p[0], p[1], p[2], c=[colour_basis[a_idx]], alpha=0.3, s=5)

    # then plot omegas as scatter points on separate axes
    for a_idx in range(n_agents):
        point = omegas[a_idx] / np.linalg.norm(omegas[a_idx])
        axs[1].scatter(point[0], point[1], point[2], c=[colour_basis[a_idx]], s=100)
    axs[1].set(xlim=[0, 1], ylim=[0, 1], zlim=[0, 1])

    # plot loss and eval data
    sns.lineplot(x=loss_data[0], y=loss_data[1], ax=axs[2], color="black")
    sns.lineplot(x=eval_data[0], y=eval_data[1], ax=axs[3], color="black")

    # set view angles, remove axis labels and set titles
    for i, ax_title in enumerate(["Featuriser", "Omegas", "Loss", "Eval"]):
        if i < 2:
            axs[i].view_init(30, 45)
            axs[i].set(xticklabels=[], yticklabels=[], zticklabels=[])
        axs[i].set_title(ax_title)
    fig.suptitle(title)

    return fig, axs


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
    visualise_interval=1000,
):
    f_optim = optax.adam(1e-2)
    f_state = f_optim.init(eqx.filter(featuriser, eqx.is_array))

    o_optim = optax.adam(1e-3)
    o_state = o_optim.init(omegas)

    @eqx.filter_jit
    def do_train_step(featuriser, omegas, f_state, o_state, obss, next_obss, a_idxs):
        f_grad, o_grad = jax.grad(loss_fn, argnums=(0, 1))(
            featuriser, omegas, obss, next_obss, a_idxs
        )

        f_updates, f_state = f_optim.update(f_grad, f_state)
        featuriser = eqx.apply_updates(featuriser, f_updates)

        o_updates, o_state = o_optim.update(o_grad, o_state)
        omegas = eqx.apply_updates(omegas, o_updates)

        return featuriser, f_state, omegas, o_state

    loss_ts, losses, eval_ts, evals, vis_frames = [], [], [], [], []
    keys = jr.split(key, n_iter)
    for i in tqdm(range(n_iter)):
        batch_idxs = jr.randint(keys[i], (batch_size,), 0, len(train_data))
        batch = train_data[batch_idxs]

        if i % log_interval == 0:
            loss = loss_fn(
                featuriser, omegas, batch.obs, batch.next_obs, batch.agent_idxs
            )
            loss_ts.append(i)
            losses.append(float(loss))
            # tqdm.write(f"loss: {loss}")

        if i % eval_interval == 0:
            eval_score = do_eval(keys[i], test_data, featuriser, omegas, n=100)
            eval_ts.append(i)
            evals.append(eval_score)
            # tqdm.write(f"eval: {eval_score}")

        if i % visualise_interval == 0:
            fig, _ = visualise_training_progress(
                featuriser,
                omegas,
                test_data,
                (loss_ts, losses),
                (eval_ts, evals),
                title=f"{i} iterations",
            )
            vis_frames.append(fig_to_pil_image(fig))
            images_to_gif(vis_frames, "training.gif")

        featuriser, f_state, omegas, o_state = do_train_step(
            featuriser,
            omegas,
            f_state,
            o_state,
            batch.obs,
            batch.next_obs,
            batch.agent_idxs,
        )

    return featuriser, omegas, losses, evals, vis_frames


# set key for jax.random
seed = int(time.time())
key = jr.PRNGKey(seed)

# sample expert transitions
env = gym.make(env_id, **env_kwargs)
# train_data = generate_expert_data(env, int(1e5))
# test_data = generate_expert_data(env, int(1e4))
policy_funcs = [expert_policy for _ in range(3)]
policy_func_kwargs = [{"fruit_prefs": fruit_prefs[m]} for m in range(3)]
print(f"generating train data...")
train_data = generate_split_dataset(
    env,
    seed,
    int(1e5),
    policy_funcs,
    policy_func_kwargs,
)
print(len(train_data))
print(f"generating test data...")
test_data = generate_split_dataset(
    env,
    seed,
    int(1e4),
    policy_funcs,
    policy_func_kwargs,
)
print(len(test_data))

# find network initialisation with lowest starting loss
omegas = jr.multivariate_normal(key, jnp.ones(3) / 3, jnp.eye(3) / 20, (3,))
best_init_loss, best_featuriser, best_omegas, best_key = jnp.inf, None, None, None
for k in tqdm(jr.split(key, 50)):
    # omegas = jr.normal(k, (3, 3))
    featuriser = MLP(
        k,
        in_size=49,
        hidden_size=128,
        out_size=3,
        num_hidden_layers=3,
    )
    loss = loss_fn(
        featuriser, omegas, train_data.obs, train_data.next_obs, train_data.agent_idxs
    )
    if loss < best_init_loss:
        best_init_loss = loss
        best_featuriser = featuriser
        best_omegas = omegas
        best_key = k
print(f"best init loss: {best_init_loss}")
featuriser, omegas, key = best_featuriser, best_omegas, best_key

# train network
featuriser, omegas, losses, evals, vis_frames = train(
    key,
    featuriser,
    omegas,
    train_data,
    test_data,
    n_iter=int(5e4),
    batch_size=int(1e4),
    log_interval=200,
    eval_interval=200,
    visualise_interval=200,
)

# plot training curves
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.lineplot(x=range(len(losses)), y=losses, ax=axs[0])
sns.lineplot(x=range(len(evals)), y=evals, ax=axs[1])
fig.savefig("training_curves.png")
