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
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
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


algos = {"ppo": PPO, "dqn": DQN}
hyperparam_defaults = {"ppo": PPO_DEFAULT_HYPERPARAMS, "dqn": DQN_DEFAULT_HYPERPARAMS}


class RLEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        resume: bool = False,
    ):
        super().__init__(
            eval_env,
            callback_on_new_best,
            callback_after_eval,
            n_eval_episodes,
            eval_freq,
            log_path,
            best_model_save_path,
            deterministic,
            render,
            verbose,
            warn,
        )
        self.evaluations_results: List[List[float]] = []
        self.evaluations_timesteps: List[int] = []
        self.evaluations_length: List[List[int]] = []
        self.consumed_counts: List[List[List[int]]] = []
        self._prev_last_timestep = 0

        if resume:
            npz_file = np.load(f"{self.log_path}.npz")
            self.evaluations_results = npz_file["results"].tolist()
            self.evaluations_timesteps = npz_file["timesteps"].tolist()
            self.evaluations_length = npz_file["ep_lengths"].tolist()
            self.consumed_counts = npz_file["consumed_counts"].tolist()
            self._prev_last_timestep = self.evaluations_timesteps[-1]

    def _on_step(self) -> bool:
        continue_training = True
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

            episode_rewards, episode_lengths, episode_consumed_counts = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                warn=self.warn,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                assert isinstance(episode_consumed_counts, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                self.consumed_counts.append(episode_consumed_counts)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    consumed_counts=self.consumed_counts,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


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


def enjoy_rl_agent(
    run_name: str,
    env_id: str,
    env_kwargs: Dict,
    algo_name: str,
    seed: int,
    checkpoint: str = "best",
    log_dir: str = "../checkpoints",
) -> None:
    assert algo_name in algos, f"Algorithm {algo_name} not supported"

    env = make_vec_env(env_id, n_envs=1, seed=seed, env_kwargs=env_kwargs)
    model = algos[algo_name].load(
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
