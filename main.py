import time

import gymnasium as gym
import numpy as np

from selective_imitation_learning.environments.fruitworld import FruitWorld

seed = 0
env = gym.make(
    "FruitWorld-v0",
    grid_size=5,
    fruits_per_type=1,
    preferences=np.array([0.8, 0.1, 0.1]),
    render_mode="human",
)
obs, _ = env.reset(seed=seed)

for t in range(10):
    env.render()
    print(t % 4)
    env.step(t % 4)
    time.sleep(1)
