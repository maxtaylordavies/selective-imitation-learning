from datetime import datetime
import os

import numpy as np

from selective_imitation_learning.learning import (
    train_rl_agent,
    plot_eval_curves,
    enjoy_rl_agent,
)

# run_name = f"ppo_{datetime.now().strftime('%Y%m%d%H%M%S')}"
algo = "ppo"
run_name = "test"
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
model_kwargs = {"learning_rate": 5e-4}

train_rl_agent(
    run_name=run_name,
    env_id=env_id,
    env_kwargs=env_kwargs,
    algo_name=algo,
    model_kwargs=model_kwargs,
    train_steps=int(5e6),
    resume=False,
)

plot_eval_curves(run_name, window_size=100)

enjoy_rl_agent(
    run_name=run_name,
    env_id=env_id,
    env_kwargs=env_kwargs,
    algo_name=algo,
    seed=seed,
    checkpoint="best",
)
