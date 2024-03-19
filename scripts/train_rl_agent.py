from datetime import datetime
import os

import numpy as np

from selective_imitation_learning.rl import (
    train_rl_agent,
    plot_eval_curves,
    demonstrate_ppo_agent,
)

# run_name = f"ppo_{datetime.now().strftime('%Y%m%d%H%M%S')}"
run_name = "test"
seed = 0
env_id = "FruitWorld-v0"
env_kwargs = {
    "grid_size": 7,
    "num_fruit": 5,
    "fruit_preferences": np.array([0.7, 0.2, 0.1]),
    # "fruit_type_probs": np.array([1.0, 0.0, 0.0]),
    "fruit_loc_means": np.array([[1, 1], [1, 5], [5, 3]]),
    "fruit_loc_stds": 1.5 * np.ones(3),
    "max_steps": 50,
    "render_mode": "human",
}
model_kwargs = {"learning_rate": 5e-4}

train_rl_agent(
    run_name=run_name,
    env_id=env_id,
    env_kwargs=env_kwargs,
    algo_name="ppo",
    model_kwargs=model_kwargs,
    train_steps=int(1e5),
    resume=False,
)

plot_eval_curves(run_name, window_size=10)

demonstrate_ppo_agent(
    run_name=run_name,
    env_id=env_id,
    env_kwargs=env_kwargs,
    seed=seed,
    checkpoint="best",
)
