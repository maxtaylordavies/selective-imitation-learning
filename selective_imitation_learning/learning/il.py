import os
from typing import Dict, Optional

from imitation.algorithms import bc
from imitation.data import types, rollout
from imitation.policies.serialize import load_policy
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm

from .utils import generate_demo_transitions
from ..rl.callbacks import evaluate_policy


class ILEvalCallback:
    def __init__(self, eval_env, eval_freq, n_eval_episodes, batch_size):
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.batch_size = batch_size
        self.steps_seen = 0
        self.last_eval = 0

    def on_batch_end(self, model):
        self.steps_seen += self.batch_size
        if self.steps_seen - self.last_eval >= self.eval_freq:
            self.last_eval = self.steps_seen
            self.run_eval(model)

    def run_eval(self, model):
        rewards, _, _ = evaluate_policy(model, self.eval_env, self.n_eval_episodes)
        tqdm.write(f"{self.steps_seen} | {np.mean(rewards)} +/- {np.std(rewards)}")


def train_bc_agent(
    run_name: str,
    env_id: str,
    env_kwargs: Dict,
    expert_model_path: str,
    expert_algo: str = "ppo",
    min_timesteps: Optional[int] = None,
    min_episodes: Optional[int] = None,
    train_steps: int = int(1e6),
    n_training_envs: int = 16,
    n_eval_envs: int = 10,
    train_seed: int = 0,
    eval_seed: int = 0,
):
    # create run directory if it doesn't exist
    # run_dir = os.path.join(log_dir, run_name)
    # os.makedirs(run_dir, exist_ok=True)

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
        eval_freq=1000,
        n_eval_episodes=10,
        batch_size=trainer.batch_size,
    )

    # run initial eval
    callback.run_eval(trainer.policy)

    # train behavior cloning model
    trainer.train(
        n_epochs=10,
        on_batch_end=lambda: callback.on_batch_end(trainer.policy),
        # progress_bar=False,
        log_interval=np.inf,
    )
