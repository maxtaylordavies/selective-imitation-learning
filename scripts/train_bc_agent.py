from datetime import datetime
import os

import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env

from selective_imitation_learning.learning import (
    generate_demonstrations,
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
    f"../checkpoints/ppo/{colour}/best_model.zip" for colour in ("red", "green", "blue")
]
expert_info = [{"preference": "red"}, {"preference": "green"}, {"preference": "blue"}]


# generate expert demonstrations
print("generating demonstration data...")
rng = np.random.default_rng(seed)
train_env = make_vec_env(env_id, n_envs=16, seed=seed, env_kwargs=env_kwargs)
demonstrations = generate_demonstrations(
    train_env,
    rng,
    expert_paths,
    expert_info,
    int(1e5),
)


# define weighting function
def weight_fn(infos):
    f = lambda info: int(info["preference"] == "blue")
    return np.array([f(info) for info in infos])


print("beginning training")
train_bc_agent(
    rng=rng,
    run_name=run_name,
    env_id=env_id,
    env_kwargs=env_kwargs,
    demonstrations=demonstrations,
    weight_fn=weight_fn,
    train_epochs=5,
)

plot_eval_curves(run_name, window_size=20)

# load trained policy
policy = load_policy(os.path.join("../checkpoints", run_name, "best_model.zip"))
enjoy_policy(policy, env_id, env_kwargs, seed=seed)
