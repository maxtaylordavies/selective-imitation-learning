import os
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback as _EvalCallback
from stable_baselines3.common.vec_env import VecEnv
from tqdm import tqdm

from ..utils import save_policy
from .utils import evaluate_policy


class EvalCallback(_EvalCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        resume: bool = False,
    ):
        super().__init__(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=log_path,
            deterministic=True,
            render=render,
            verbose=verbose,
            warn=warn,
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

    def _run_eval(self):
        episode_rewards, episode_lengths, episode_consumed_counts = evaluate_policy(
            self.model.policy,
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
            tqdm.write(
                f"{self.num_timesteps} timesteps | {mean_reward:.2f} +/- {std_reward:.2f}"
            )

        if mean_reward > self.best_mean_reward:
            if self.verbose >= 1:
                tqdm.write("New best mean reward!")
            self._save_best_model()
            self.best_mean_reward = float(mean_reward)

    def _save_best_model(self):
        if self.best_model_save_path is None:
            return
        save_policy(
            self.model.policy, os.path.join(self.best_model_save_path, "best_model")
        )

    # def _save_best_model(self):
    #     if self.best_model_save_path is None:
    #         return
    #     self.model.save(os.path.join(self.best_model_save_path, "best_model"))
