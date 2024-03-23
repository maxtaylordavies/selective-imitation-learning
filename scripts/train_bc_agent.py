from datetime import datetime
import os

import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy

from selective_imitation_learning.learning import (
    train_bc_agent,
    plot_eval_curves,
    load_policy,
    enjoy_policy,
)

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


expert_paths = [
    f"../checkpoints/ppo/{colour}/best_model.zip" for colour in ("blue", "red")
]
train_bc_agent(
    run_name=run_name,
    env_id=env_id,
    env_kwargs=env_kwargs,
    expert_model_paths=expert_paths,
    min_timesteps=int(1e5),
    train_epochs=10,
)

plot_eval_curves(run_name, window_size=20)

# load trained policy
policy = load_policy(os.path.join("../checkpoints", run_name, "best_model.zip"))
enjoy_policy(policy, env_id, env_kwargs, seed=seed)
