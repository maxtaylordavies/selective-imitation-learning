import os
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from selective_imitation_learning.environments.fruitworld import FruitWorld
from selective_imitation_learning.constants import PPO_DEFAULT_HYPERPARAMS


def train_ppo_agent(
    run_name: str,
    env_id: str,
    env_kwargs: Dict,
    model_kwargs: Optional[Dict] = None,
    train_steps: int = int(1e6),
    n_training_envs: int = 8,
    n_eval_envs: int = 5,
    train_seed: int = 0,
    eval_seed: int = 0,
    eval_freq: int = 1000,
    n_eval_episodes: int = 10,
    log_dir: str = "../checkpoints",
):
    # create training and evaluation environments
    train_env = make_vec_env(
        env_id, n_envs=n_training_envs, seed=train_seed, env_kwargs=env_kwargs
    )
    eval_env = make_vec_env(
        env_id, n_envs=n_eval_envs, seed=eval_seed, env_kwargs=env_kwargs
    )

    # create model
    hyperparams = PPO_DEFAULT_HYPERPARAMS.copy()
    if model_kwargs is not None:
        hyperparams.update(model_kwargs)
    model = PPO(
        "MlpPolicy",
        train_env,
        **hyperparams,
    )

    # create eval callback
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    # train model
    model.learn(total_timesteps=train_steps, progress_bar=True, callback=eval_callback)

    # save model
    model.save(os.path.join(run_dir, "final_model"))
