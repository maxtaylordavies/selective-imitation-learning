from datetime import datetime
import os

import numpy as np

from selective_imitation_learning.rl import (
    train_dqn_agent,
    plot_eval_curves,
    # demonstrate_ppo_agent,
)

# run_name = f"ppo_{datetime.now().strftime('%Y%m%d%H%M%S')}"
run_name = "test_dqn"
seed = 0
env_id = "FruitWorld-v0"
env_kwargs = {
    "grid_size": 5,
    "fruits_per_type": 1,
    "preferences": np.array([1.0, 0.0, 0.0]),
    "fruit_loc_means": np.array([[0, 0], [0, 4], [4, 2]]),
    # "fruit_loc_stds": np.array([0.5, 0.5, 0.5]),
    "max_steps": 50,
    "render_mode": "human",
}
model_kwargs = {
    "learning_rate": 0.0000625,
    "batch_size": 64,
    "gamma": 0.99,
    "exploration_initial_eps": 0.9,
    "exploration_final_eps": 0.01,
    "target_update_interval": 80,
    "learning_starts": 10000,
    "train_freq": 4,
}

train_dqn_agent(
    run_name=run_name,
    env_id=env_id,
    env_kwargs=env_kwargs,
    train_steps=int(1e7),
    eval_freq=20000,
    resume=False,
)

plot_eval_curves(run_name)
