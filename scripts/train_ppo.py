from datetime import datetime
import os

import numpy as np

from selective_imitation_learning.rl import (
    train_ppo_agent,
    plot_reward_curve,
    demonstrate_ppo_agent,
)

run_name = f"ppo_{datetime.now().strftime('%Y%m%d%H%M%S')}"
seed = 0
env_id = "FruitWorld-v0"
env_kwargs = {
    "grid_size": 7,
    "fruits_per_type": 1,
    "preferences": np.array([0.8, 0.1, 0.1]),
    "max_steps": 30,
    "render_mode": "human",
}

train_ppo_agent(
    run_name=run_name,
    env_id=env_id,
    env_kwargs=env_kwargs,
    train_steps=int(5e6),
)

plot_reward_curve(run_name)

demonstrate_ppo_agent(
    run_name=run_name,
    env_id=env_id,
    env_kwargs=env_kwargs,
    seed=seed,
)
