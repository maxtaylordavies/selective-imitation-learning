import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from selective_imitation_learning.constants import ENV_CONSTANTS

sns.set_theme()


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


def demonstrate_ppo_agent(
    run_name: str,
    env_id: str,
    env_kwargs: Dict,
    seed: int,
    checkpoint: str = "best",
    log_dir: str = "../checkpoints",
) -> None:
    env = make_vec_env(env_id, n_envs=1, seed=seed, env_kwargs=env_kwargs)

    model = PPO.load(
        os.path.join(log_dir, run_name, f"{checkpoint}_model.zip"), env=env
    )
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, infos = env.step(action)
        for info in infos:
            if "consumed_counts" in info.keys():
                print(f"Consumed counts: {info['consumed_counts']}")
        env.render("human")
