import os
from typing import Any, Callable, Dict, Optional, List, Union, Tuple
import warnings

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env

from selective_imitation_learning.environments.fruitworld import FruitWorld
from selective_imitation_learning.constants import (
    PPO_DEFAULT_HYPERPARAMS,
    DQN_DEFAULT_HYPERPARAMS,
)
from .callbacks import CustomEvalCallback

algos = {"ppo": PPO, "dqn": DQN}
hyperparam_defaults = {"ppo": PPO_DEFAULT_HYPERPARAMS, "dqn": DQN_DEFAULT_HYPERPARAMS}


def train_rl_agent(
    run_name: str,
    env_id: str,
    env_kwargs: Dict,
    algo_name: str,
    model_kwargs: Optional[Dict] = None,
    train_steps: int = int(1e6),
    n_training_envs: int = 16,
    n_eval_envs: int = 10,
    train_seed: int = 0,
    eval_seed: int = 0,
    eval_freq: int = 5000,
    n_eval_episodes: int = 20,
    log_dir: str = "../checkpoints",
    resume: bool = False,
):
    assert algo_name in algos, f"Algorithm {algo_name} not supported"

    # create run directory if it doesn't exist
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # create training and evaluation environments
    train_env = make_vec_env(
        env_id, n_envs=n_training_envs, seed=train_seed, env_kwargs=env_kwargs
    )
    eval_env = make_vec_env(
        env_id, n_envs=n_eval_envs, seed=eval_seed, env_kwargs=env_kwargs
    )

    # load or create model
    if resume:
        model = algos[algo_name].load(os.path.join(run_dir, "final_model"), train_env)
    else:
        hparams = hyperparam_defaults[algo_name].copy()
        if model_kwargs is not None:
            hparams.update(model_kwargs)
        model = algos[algo_name]("MlpPolicy", train_env, **hparams)

    # create eval callback
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(eval_freq // n_training_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        resume=resume,
    )

    # train model
    model.learn(total_timesteps=train_steps, progress_bar=True, callback=eval_callback)

    # save model
    model.save(os.path.join(run_dir, "final_model"))
