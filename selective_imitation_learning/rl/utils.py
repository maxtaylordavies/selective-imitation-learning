import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

sns.set_theme()


def plot_reward_curve(
    run_name: str, log_dir: str = "../checkpoints", window_size: int = 5
) -> None:
    npzfile = np.load(os.path.join(log_dir, run_name, "evaluations.npz"))
    eval_returns = npzfile["results"]
    eval_steps = npzfile["timesteps"]

    data = {"timestep": [], "return": [], "env": []}
    for i, step in enumerate(eval_steps):
        for j, ret in enumerate(eval_returns[i]):
            data["timestep"].append(step)
            data["return"].append(ret)
            data["env"].append(j)
    df = pd.DataFrame(data)

    df["return"] = df["return"].rolling(window_size).mean()
    sns.lineplot(data=df, x="timestep", y="return")
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
        obs, _, _, _ = env.step(action)
        env.render("human")
