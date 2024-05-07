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


from selective_imitation_learning.environments.ma_fruitworld import expert_policy
from selective_imitation_learning.data import (
    UnifiedMultiAgentTransitions,
    generate_unified_dataset,
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
env_id = "MultiAgentFruitWorld-v0"
env_kwargs = {
    "grid_size": 7,
    "num_fruit": 3,
    "fruit_prefs": fruit_prefs,
    "fruit_loc_means": np.array([[1, 2], [2, 5], [4, 3]]),
    "fruit_loc_stds": 1.5 * np.ones(3),
    "max_steps": 50,
    "render_mode": "human",
}


@jax.jit
def perspective_fn(obs: jax.Array) -> jax.Array:
    agent_map, fruit_map = obs[0], obs[1]
    # M = len(jnp.unique(agent_map)) - 1
    ps = jnp.tile(fruit_map, (3, 1, 1))
    for m in range(3):
        pos = jnp.where(agent_map == m + 1, size=1)
        ps = ps.at[m, pos[0][0], pos[1][0]].set(-1.0)
    return ps


@jax.jit
def transition_ll(delta_fs: jax.Array, omegas: jax.Array) -> jax.Array:
    delta_fs = jnp.array([to_simplex(delta_f) for delta_f in delta_fs])
    return jnp.sum(delta_fs * omegas)


@eqx.filter_jit
def loss_fn(p_func, f_func, omegas, obss, next_obss):
    # pass observations through perspective function
    # output has shape (N, M, p_shape)
    ps = jax.vmap(p_func)(obss)
    p_nexts = jax.vmap(p_func)(next_obss)

    # pass perspectives through featuriser
    # output has shape (N, M, num_features)
    fs = jax.vmap(jax.vmap(f_func))(ps)
    f_nexts = jax.vmap(jax.vmap(f_func))(p_nexts)
    delta_fs = f_nexts - fs

    # compute log likelihood of transitions
    _omegas = jnp.array([to_simplex(omega) for omega in omegas])
    lls = jax.vmap(transition_ll, in_axes=(0, None))(delta_fs, _omegas)

    # compute omegas penalty
    tmp = jnp.dot(_omegas, _omegas.T)
    o_penalty = tmp.sum() - jnp.trace(tmp)
    o_penalty *= 0.1 * len(lls)

    return -jnp.sum(lls) + o_penalty


def sample_episodes(key, transitions, n, ep_len=50):
    ep_start_idxs = jnp.where(transitions.dones)[0] + 1
    ep_start_idxs = jnp.concatenate([jnp.array([0]), ep_start_idxs[:-1]])
    starts = jax.random.choice(key, ep_start_idxs, shape=(n,), replace=False)

    eps = []
    for start in starts:
        eps.append(transitions[start : start + ep_len])
    return eps


def compute_agent_probs(p_func, f_func, omegas, episode):
    ps = jax.vmap(p_func)(episode.obs)
    p_nexts = jax.vmap(p_func)(episode.next_obs)

    fs = jax.vmap(jax.vmap(f_func))(ps)
    f_nexts = jax.vmap(jax.vmap(f_func))(p_nexts)
    delta_fs = f_nexts - fs

    delta_f_sums = np.zeros((3, 3))
    agent_probs = np.zeros((3, 3))
    for i in range(len(episode)):
        for m in range(3):
            df = to_simplex(delta_fs[i][m])
            delta_f_sums[m] += df
            for n in range(3):
                agent_probs[m][n] += float(df @ to_simplex(omegas[n]))
            # agent_probs[m] += float(df @ to_simplex(omegas[m]))

    agent_probs = jnp.exp(jnp.array(agent_probs))
    agent_probs = jnp.array([to_simplex(agent_probs[m]) for m in range(3)])

    return agent_probs, delta_f_sums / len(episode)


def do_eval(key, p_func, f_func, transitions, omegas, n=100, print_fs=False):
    episodes, score = sample_episodes(key, transitions, n), 0
    for ep in episodes:
        agent_probs, _ = compute_agent_probs(p_func, f_func, omegas, ep)
        argmaxes = jnp.argmax(agent_probs, axis=1)
        score += int(jnp.sum(argmaxes == jnp.arange(3)))
    return score / (n * 3)


def featuriser_histogram(p_func, f_func, omegas, episodes, bins=125):
    hists, fss = [], []
    side = round(bins ** (1 / 3))
    for m in range(3):
        hists.append(np.zeros((side, side, side)))
        fss.append([])

    for ep in episodes:
        _, df_means = compute_agent_probs(p_func, f_func, omegas, ep)
        for m in range(3):
            fs = df_means[m] / np.linalg.norm(df_means[m])
            fss[m].append(fs * side)
            idxs = np.clip(np.floor(fs * side).astype(np.int32), 0, side - 1)
            hists[m][idxs[0], idxs[1], idxs[2]] += 1

    return hists, fss


def visualise_training_progress(
    p_func,
    f_func,
    omegas,
    transitions,
    loss_data,
    eval_data,
    n_samples=100,
    bins=1000,
    title="",
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
    n_agents = omegas.shape[0]
    histograms, scatter_points = np.zeros((side, side, side, n_agents)), []

    eps = sample_episodes(jr.PRNGKey(0), transitions, n_samples)
    hists, points = featuriser_histogram(p_func, f_func, omegas, eps, bins=bins)
    for m in range(n_agents):
        histograms[..., m] = hists[m]
        scatter_points.append(points[m])

    # now transform hists into an array of rgb values
    # each bin should be mapped to a colour, which is a
    # weighted sum of the 3 fruit colours (weighted by the bin value per agent)
    colour_basis = np.array(ENV_CONSTANTS["fruit_colours"]) / 255
    rgb = np.zeros((side, side, side, 3))
    for m in range(n_agents):
        rgb += histograms[..., m, None] * colour_basis[m]

    # normalise colours
    histograms_sum = np.sum(histograms, axis=-1)
    tmp = histograms_sum.copy()
    tmp[tmp == 0] = 1
    rgb /= tmp[..., None]

    # use the sum of the histograms to set the alpha channel
    alphas = histograms_sum / np.max(histograms_sum)
    rgba = np.concatenate([rgb, alphas[..., None]], axis=-1)

    # plot
    axs[0].voxels(alphas > 0, facecolors=rgba, edgecolors=rgba)
    for m, pts in enumerate(points):
        for p in pts:
            axs[0].scatter(p[0], p[1], p[2], c=[colour_basis[m]], alpha=0.3, s=5)

    # then plot omegas as scatter points on separate axes
    for m in range(n_agents):
        point = omegas[m] / np.linalg.norm(omegas[m])
        axs[1].scatter(point[0], point[1], point[2], c=[colour_basis[m]], s=100)
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
    p_func,
    f_func,
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
    f_state = f_optim.init(eqx.filter(f_func, eqx.is_array))

    o_optim = optax.adam(1e-3)
    o_state = o_optim.init(omegas)

    @eqx.filter_jit
    def do_train_step(f_func, omegas, f_state, o_state, obss, next_obss):
        f_grad, o_grad = jax.grad(loss_fn, argnums=(1, 2))(
            p_func, f_func, omegas, obss, next_obss
        )

        f_updates, f_state = f_optim.update(f_grad, f_state)
        f_func = eqx.apply_updates(f_func, f_updates)

        o_updates, o_state = o_optim.update(o_grad, o_state)
        omegas = eqx.apply_updates(omegas, o_updates)

        return f_func, f_state, omegas, o_state

    loss_ts, losses, eval_ts, evals, vis_frames = [], [], [], [], []
    keys = jr.split(key, n_iter)

    # initial visualisation
    fig, _ = visualise_training_progress(
        p_func,
        f_func,
        omegas,
        test_data,
        (loss_ts, losses),
        (eval_ts, evals),
        title="Initial",
    )
    fig.savefig("initial.svg")

    # main training loop
    for i in tqdm(range(n_iter)):
        batch_idxs = jr.randint(keys[i], (batch_size,), 0, len(train_data))
        batch = train_data[batch_idxs]

        if i % log_interval == 0:
            loss = loss_fn(p_func, f_func, omegas, batch.obs, batch.next_obs)
            loss_ts.append(i)
            losses.append(float(loss))
            tqdm.write(f"loss: {loss}")

        if i % eval_interval == 0:
            eval_score = do_eval(keys[i], p_func, f_func, test_data, omegas, n=100)
            eval_ts.append(i)
            evals.append(eval_score)
            # tqdm.write(f"eval: {eval_score}")

        if i % visualise_interval == 0:
            fig, _ = visualise_training_progress(
                p_func,
                f_func,
                omegas,
                test_data,
                (loss_ts, losses),
                (eval_ts, evals),
                title=f"{i} iterations",
            )
            vis_frames.append(fig_to_pil_image(fig))
            images_to_gif(vis_frames, "training.gif")

        f_func, f_state, omegas, o_state = do_train_step(
            f_func,
            omegas,
            f_state,
            o_state,
            batch.obs,
            batch.next_obs,
        )

    # final visualisation
    fig, _ = visualise_training_progress(
        p_func,
        f_func,
        omegas,
        test_data,
        (loss_ts, losses),
        (eval_ts, evals),
        title="Initial",
    )
    fig.savefig("final.svg")

    return f_func, omegas, losses, evals, vis_frames


# set key for jax.random
seed = int(time.time())
key = jr.PRNGKey(seed)

# sample expert transitions
env = gym.make(env_id, **env_kwargs)
print(f"generating train data...")
train_data = generate_unified_dataset(
    env, seed, int(1e6), expert_policy, {"fruit_prefs": fruit_prefs}
)
print(len(train_data))
print(f"generating test data...")
test_data = generate_unified_dataset(
    env, seed, int(3e4), expert_policy, {"fruit_prefs": fruit_prefs}
)
print(len(test_data))

# find network initialisation with lowest starting loss
best_init_loss, best_featuriser, best_omegas, best_key = jnp.inf, None, None, None
for k in tqdm(jr.split(key, 50)):
    omegas = jr.multivariate_normal(k, jnp.ones(3) / 3, jnp.eye(3) / 20, (3,))
    featuriser = MLP(
        k,
        in_size=len(perspective_fn(train_data.obs[0])[0].flatten()),
        hidden_size=128,
        out_size=3,
        num_hidden_layers=3,
    )
    subset = train_data[jr.randint(k, (int(1e4),), 0, len(train_data))]
    loss = loss_fn(perspective_fn, featuriser, omegas, subset.obs, subset.next_obs)
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
    perspective_fn,
    featuriser,
    omegas,
    train_data,
    test_data,
    n_iter=int(5e4),
    batch_size=int(2e3),
    log_interval=100,
    eval_interval=200,
    visualise_interval=200,
)
