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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
    sync_envs_normalization,
)
from stable_baselines3.common.logger import Logger

from selective_imitation_learning.environments.fruitworld import FruitWorld
from selective_imitation_learning.constants import (
    PPO_DEFAULT_HYPERPARAMS,
    DQN_DEFAULT_HYPERPARAMS,
)
from .callback import EvalCallback
from .utils import save_policy


algos = {"ppo": PPO, "dqn": DQN}
hyperparam_defaults = {"ppo": PPO_DEFAULT_HYPERPARAMS, "dqn": DQN_DEFAULT_HYPERPARAMS}


class RLEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        self.num_timesteps += self._prev_last_timestep
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e
            self._run_eval()
        return True


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
    eval_freq: int = 20000,
    n_eval_episodes: int = 50,
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
    eval_callback = RLEvalCallback(
        eval_env,
        log_path=run_dir,
        eval_freq=max(eval_freq // n_training_envs, 1),
        n_eval_episodes=n_eval_episodes,
        resume=resume,
    )

    # train model
    model.learn(total_timesteps=train_steps, progress_bar=True, callback=eval_callback)

    # save final model
    save_policy(model.policy, os.path.join(run_dir, "final_model"))
