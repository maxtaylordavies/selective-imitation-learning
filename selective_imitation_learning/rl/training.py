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
from selective_imitation_learning.constants import PPO_DEFAULT_HYPERPARAMS
from .callbacks import CustomEvalCallback


def train_ppo_agent(
    run_name: str,
    env_id: str,
    env_kwargs: Dict,
    model_kwargs: Optional[Dict] = None,
    train_steps: int = int(1e6),
    n_training_envs: int = 16,
    n_eval_envs: int = 5,
    train_seed: int = 0,
    eval_seed: int = 0,
    eval_freq: int = 5000,
    n_eval_episodes: int = 20,
    log_dir: str = "../checkpoints",
    resume: bool = False,
):
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # create training and evaluation environments
    train_env = make_vec_env(
        env_id, n_envs=n_training_envs, seed=train_seed, env_kwargs=env_kwargs
    )
    eval_env = make_vec_env(
        env_id, n_envs=n_eval_envs, seed=eval_seed, env_kwargs=env_kwargs
    )

    # create or load model
    hyperparams = PPO_DEFAULT_HYPERPARAMS.copy()
    if model_kwargs is not None:
        hyperparams.update(model_kwargs)
    model = (
        PPO.load(os.path.join(run_dir, "final_model"), train_env)
        if resume
        else PPO(
            "MlpPolicy",
            train_env,
            **hyperparams,
        )
    )

    # create eval callback
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(eval_freq // n_eval_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        resume=resume,
    )

    # train model
    model.learn(total_timesteps=train_steps, progress_bar=True, callback=eval_callback)

    # save model
    model.save(os.path.join(run_dir, "final_model"))


def train_dqn_agent(
    run_name: str,
    env_id: str,
    env_kwargs: Dict,
    model_kwargs: Optional[Dict] = None,
    train_steps: int = int(1e6),
    n_training_envs: int = 16,
    n_eval_envs: int = 5,
    train_seed: int = 0,
    eval_seed: int = 0,
    eval_freq: int = 5000,
    n_eval_episodes: int = 20,
    log_dir: str = "../checkpoints",
    resume: bool = False,
):
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # create training and evaluation environments
    train_env = make_vec_env(
        env_id, n_envs=n_training_envs, seed=train_seed, env_kwargs=env_kwargs
    )
    eval_env = make_vec_env(
        env_id, n_envs=n_eval_envs, seed=eval_seed, env_kwargs=env_kwargs
    )

    # create or load model
    model_kwargs = model_kwargs or {}
    model = (
        DQN("MlpPolicy", train_env, **model_kwargs)
        if not resume
        else DQN.load(os.path.join(run_dir, "final_model"), train_env)
    )

    # create eval callback
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(eval_freq // n_eval_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        resume=resume,
    )

    # train model
    model.learn(total_timesteps=train_steps, progress_bar=True, callback=eval_callback)

    # save model
    model.save(os.path.join(run_dir, "final_model"))
