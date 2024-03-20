from datetime import datetime
import os

import numpy as np

from selective_imitation_learning.il.training import (
    train_bc_agent,
)

# run_name = f"ppo_{datetime.now().strftime('%Y%m%d%H%M%S')}"
run_name = "test_bc"
seed = 0
env_id = "FruitWorld-v0"
env_kwargs = {
    "grid_size": 5,
    "num_fruit": 3,
    "fruit_preferences": np.array([0.0, 0.0, 1.0]),
    "fruit_loc_means": np.array([[0, 0], [0, 4], [4, 2]]),
    "fruit_loc_stds": 1 * np.ones(3),
    "max_steps": 50,
    "render_mode": "human",
}


train_bc_agent(
    run_name=run_name,
    env_id=env_id,
    env_kwargs=env_kwargs,
    expert_model_path="../checkpoints/ppo/blue/best_model.zip",
    min_timesteps=int(1e5),
)
