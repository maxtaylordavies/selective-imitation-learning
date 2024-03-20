import os
from struct import Struct
from typing import Dict, Optional, Union

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from imitation.algorithms import bc
from imitation.data import types, rollout
from imitation.policies.serialize import load_policy
from imitation.policies import base as policy_base
from imitation.util.util import save_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

from .utils import generate_demo_transitions
from .callback import EvalCallback


def save_il_model(model: BasePolicy, path: str):
    pytorch_variables = {"policy": model}
    params_to_save = {"policy": model.state_dict()}
    save_to_zip_file(path, params=params_to_save, pytorch_variables=pytorch_variables)


class ILEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        batch_size: int,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
    ):
        super().__init__(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
        )
        self.batch_size = batch_size
        self.num_timesteps = 0
        self.last_eval = 0

    def set_model(self, model):
        self.model = model

    def on_batch_end(self, model):
        self.set_model(model)
        self.num_timesteps += self.batch_size
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            self._run_eval()

    def _save_best_model(self):
        if self.best_model_save_path is None:
            return
        save_il_model(self.model, os.path.join(self.best_model_save_path, "best_model"))


def train_bc_agent(
    run_name: str,
    env_id: str,
    env_kwargs: Dict,
    expert_model_path: str,
    expert_algo: str = "ppo",
    min_timesteps: Optional[int] = None,
    min_episodes: Optional[int] = None,
    train_epochs: int = 1,
    n_training_envs: int = 16,
    n_eval_envs: int = 10,
    train_seed: int = 0,
    eval_seed: int = 0,
    log_dir: str = "../checkpoints",
):
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

    # generate expert demonstrations
    rng = np.random.default_rng(train_seed)
    transitions = generate_demo_transitions(
        train_env,
        expert_model_path,
        rng,
        min_timesteps,
        min_episodes,
        expert_algo,
    )

    trainer = bc.BC(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        demonstrations=transitions,
        rng=rng,
    )

    callback = ILEvalCallback(
        eval_env=eval_env,
        eval_freq=2000,
        n_eval_episodes=10,
        batch_size=trainer.batch_size,
        log_path=run_dir,
    )

    # run initial eval
    callback.set_model(trainer.policy)
    callback._run_eval()

    # train behavior cloning model
    trainer.train(
        n_epochs=train_epochs,
        on_batch_end=lambda: callback.on_batch_end(trainer.policy),
        # progress_bar=False,
        log_interval=np.inf,
    )

    # save final model
    save_il_model(trainer.policy, os.path.join(run_dir, "final_model"))
