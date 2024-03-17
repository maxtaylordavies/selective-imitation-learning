import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from selective_imitation_learning.environments.fruitworld import FruitWorld


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


n_training_envs = 8
n_eval_envs = 5

# Create training and evaluation environments
env_id = "FruitWorld-v0"
env_kwargs = {
    "grid_size": 7,
    "fruits_per_type": 1,
    "preferences": np.array([0.8, 0.1, 0.1]),
    "max_steps": 30,
    "render_mode": "human",
}
train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0, env_kwargs=env_kwargs)
eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0, env_kwargs=env_kwargs)

# Create eval callback
log_dir = "../checkpoints"
os.makedirs(log_dir, exist_ok=True)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=max(500 // n_training_envs, 1),
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

# Instantiate and train the agent
model = PPO(
    "MlpPolicy",
    train_env,
    normalize_advantage=True,
    gae_lambda=0.95,
    gamma=0.99,
    n_steps=128,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.0,
    learning_rate=2.5e-4,
    clip_range=0.2,
)
train_steps = int(1e5)
model.learn(total_timesteps=train_steps, progress_bar=True, callback=eval_callback)


# Load the trained agent
eval_env = make_vec_env(env_id, n_envs=1, seed=0, env_kwargs=env_kwargs)
model = PPO.load("../checkpoints/best_model", env=eval_env)

# # Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
