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
from imitation.policies.serialize import load_policy

from selective_imitation_learning.constants import ENV_CONSTANTS


def generate_demo_transitions(
    env: VecEnv,
    model_path: str,
    rng: np.random.Generator,
    min_timesteps: Optional[int] = None,
    min_episodes: Optional[int] = None,
    algo_name="ppo",
) -> types.Transitions:
    assert (
        min_timesteps is not None or min_episodes is not None
    ), "Must specify min_timesteps or min_episodes"
    policy = load_policy(algo_name, env, path=model_path)
    rollouts = rollout.rollout(
        policy,
        env,
        rollout.make_sample_until(
            min_timesteps=min_timesteps, min_episodes=min_episodes
        ),
        rng=rng,
        unwrap=False,
    )
    return rollout.flatten_trajectories(rollouts)


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
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
        actions, states = model.predict(
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


def plot_eval_curves(
    run_name: str, log_dir: str = "../checkpoints", window_size: int = 5
) -> None:
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

    # plot returns
    return_df = pd.DataFrame(return_data)
    return_df["return"] = return_df["return"].rolling(window_size).mean()
    sns.lineplot(data=return_df, x="timestep", y="return")
    plt.show()

    # plot consumed counts
    consumed_df = pd.DataFrame(consumed_data)
    consumed_df["consumed"] = consumed_df.groupby("fruit")["consumed"].transform(
        lambda x: x.rolling(window_size).mean()
    )
    colours = np.array(ENV_CONSTANTS["fruit_colours"]) / 255
    sns.lineplot(
        data=consumed_df,
        x="timestep",
        y="consumed",
        hue="fruit",
        palette=sns.color_palette(list(colours)),
    )
    plt.show()
