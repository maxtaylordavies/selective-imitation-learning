from datetime import datetime
import os

import numpy as np

from selective_imitation_learning.rl import (
    train_ppo_agent,
    plot_eval_curves,
    demonstrate_ppo_agent,
)

# run_name = f"ppo_{datetime.now().strftime('%Y%m%d%H%M%S')}"
run_name = "test"
seed = 0
env_id = "FruitWorld-v0"
env_kwargs = {
    "grid_size": 5,
    "fruits_per_type": 1,
    "preferences": np.array([1.0, 0.0, 0.0]),
    "max_steps": 50,
    "render_mode": "human",
}

train_ppo_agent(
    run_name=run_name,
    env_id=env_id,
    env_kwargs=env_kwargs,
    train_steps=int(1e5),
    resume=True,
)

plot_eval_curves(run_name)

demonstrate_ppo_agent(
    run_name=run_name,
    env_id=env_id,
    env_kwargs=env_kwargs,
    seed=seed,
    checkpoint="best",
)
