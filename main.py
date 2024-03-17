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
    max_steps=50,
    render_mode="human",
)

print(env.observation_space.shape)

obs, _ = env.reset(seed=seed)
env.render()

done = False
while not done:
    time.sleep(0.2)
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    env.render()
