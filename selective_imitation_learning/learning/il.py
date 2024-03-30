import os
from struct import Struct
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from imitation.algorithms import bc
from imitation.data import types, rollout
from imitation.policies.serialize import load_policy
from imitation.policies import base as policy_base
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

from .data import (
    MultiAgentTransitions,
    generate_demonstrations,
    make_data_loader,
)
from .callback import EvalCallback
from .utils import save_policy


class SelectiveBC(bc.BC):
    def __init__(self, *args, weight_fn: Optional[Callable] = None, **kwargs):
        self.weight_fn = weight_fn
        super().__init__(*args, **kwargs)

    def set_demonstrations(self, demonstrations: MultiAgentTransitions) -> None:
        self._demo_data_loader = make_data_loader(
            demonstrations, self.minibatch_size, weight_fn=self.weight_fn
        )


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


def train_bc_agent(
    rng: np.random.Generator,
    run_name: str,
    env_id: str,
    env_kwargs: Dict,
    demonstrations: MultiAgentTransitions,
    weight_fn: Optional[Callable] = None,
    train_epochs: int = 1,
    batch_size: int = 64,
    eval_freq: int = 10000,
    n_eval_envs: int = 10,
    n_eval_eps: int = 20,
    eval_seed: int = 0,
    log_dir: str = "../checkpoints",
):
    # create run directory if it doesn't exist
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    env = make_vec_env(
        env_id, n_envs=n_eval_envs, seed=eval_seed, env_kwargs=env_kwargs
    )

    # compute number of batches
    n_batches = int(train_epochs * (len(demonstrations) / batch_size))

    bc = SelectiveBC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=demonstrations,
        batch_size=batch_size,
        rng=rng,
        weight_fn=weight_fn,
    )

    callback = ILEvalCallback(
        eval_env=env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_eps,
        batch_size=bc.batch_size,
        log_path=run_dir,
    )

    # run initial eval
    callback.set_model(bc)
    callback._run_eval()

    # train behavior cloning model
    bc.train(
        n_batches=n_batches,
        on_batch_end=lambda: callback.on_batch_end(bc),
        # progress_bar=False,
        log_interval=np.inf,
    )

    # save final model
    save_policy(bc.policy, os.path.join(run_dir, "final_model"))
