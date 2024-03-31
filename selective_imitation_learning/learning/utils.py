import os
from typing import Any, Callable, Dict, Optional, List, Union, Tuple
import warnings

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
    sync_envs_normalization,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import Logger
from imitation.data import types, rollout
from imitation.policies.serialize import load_policy as load_expert_policy

from selective_imitation_learning.constants import ENV_CONSTANTS

sns.set_theme(style="darkgrid")


def load_eval_data(run_name: str, log_dir: str = "../checkpoints"):
    npzfile = np.load(os.path.join(log_dir, run_name, "evaluations.npz"))
    eval_returns = npzfile["results"]
    eval_steps = npzfile["timesteps"]
    consumed_counts = npzfile["consumed_counts"]

    return_data = {"timestep": [], "return": [], "env": []}
    consumed_data = {"timestep": [], "fruit": [], "consumed": [], "env": []}

    for i, step in enumerate(eval_steps):
        for j, ret in enumerate(eval_returns[i]):
            return_data["timestep"].append(step)
            return_data["return"].append(ret)
            return_data["env"].append(j)
            for k, consumed in enumerate(consumed_counts[i][j]):
                consumed_data["timestep"].append(step)
                consumed_data["fruit"].append(k)
                consumed_data["consumed"].append(consumed)
                consumed_data["env"].append(j)

    return pd.DataFrame(return_data), pd.DataFrame(consumed_data)


def take_rolling_mean_of_eval_data(
    return_df: pd.DataFrame, consumed_df: pd.DataFrame, window_size: int = 5
) -> None:
    if window_size <= 1:
        return
    return_df["return"] = return_df["return"].rolling(window_size).mean()
    consumed_df["consumed"] = consumed_df.groupby("fruit")["consumed"].transform(
        lambda x: x.rolling(window_size).mean()
    )


def plot_eval_curves(
    run_name: str, log_dir: str = "../checkpoints", window_size: int = 5
) -> None:
    return_df, consumed_df = load_eval_data(run_name=run_name, log_dir=log_dir)
    take_rolling_mean_of_eval_data(return_df, consumed_df, window_size)

    # plot returns
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    sns.lineplot(data=return_df, x="timestep", y="return", ax=axs[0])
    axs[0].set_title("Mean eval return over training")

    # plot consumed counts
    colours = np.array(ENV_CONSTANTS["fruit_colours"]) / 255
    sns.lineplot(
        data=consumed_df,
        x="timestep",
        y="consumed",
        hue="fruit",
        palette=sns.color_palette(list(colours)),
        ax=axs[1],
    )
    axs[1].set_title("Mean eval fruit consumption over training")
    fig.tight_layout()
    plt.show()


def evaluate_policy(
    policy: BasePolicy,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    warn: bool = True,
) -> Tuple[List[float], List[int], List[List[int]]]:
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]
    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    consumed_counts = []

    # divide episodes among different sub environments in the vector as evenly as possible
    episode_counts = np.zeros(n_envs, dtype="int")
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards, current_lengths = np.zeros(n_envs), np.zeros(n_envs, dtype="int")
    observations, states = env.reset(), None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    while (episode_counts < episode_count_targets).any():
        actions, states = policy.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                episode_starts[i] = dones[i]
                if dones[i]:
                    if is_monitor_wrapped:
                        if "episode" in infos[i].keys():
                            episode_rewards.append(infos[i]["episode"]["r"])
                            episode_lengths.append(infos[i]["episode"]["l"])
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1

                    current_rewards[i] = 0
                    current_lengths[i] = 0

                    if "consumed_counts" in infos[i].keys():
                        consumed_counts.append(infos[i]["consumed_counts"])

        observations = new_observations
        if render:
            env.render()

    return episode_rewards, episode_lengths, consumed_counts


def enjoy_policy(policy: BasePolicy, env_id: str, env_kwargs: Dict, seed: int):
    env = make_vec_env(env_id, n_envs=1, seed=seed, env_kwargs=env_kwargs)
    obs = env.reset()
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, _, infos = env.step(action)
        for info in infos:
            if "consumed_counts" in info.keys():
                print(f"Consumed counts: {info['consumed_counts']}")
        env.render("human")
