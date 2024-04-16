from datetime import datetime
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from selective_imitation_learning.learning import (
    train_rl_agent,
    plot_eval_curves,
    load_eval_data,
    take_rolling_mean_of_eval_data,
    load_policy,
    enjoy_policy,
)

# run_name = f"ppo_{datetime.now().strftime('%Y%m%d%H%M%S')}"
algo = "ppo"
run_name = "easy/red"
seed = 0
env_id = "FruitWorld-v0"
env_kwargs = {
    "grid_size": 5,
    "num_fruit": 3,
    "fruit_preferences": np.array([1.0, 0.0, 0.0]),
    "fruit_loc_means": np.array([[0, 0], [0, 4], [4, 2]]),
    "fruit_loc_stds": 1 * np.ones(3),
    "max_steps": 50,
    "render_mode": "human",
}
model_kwargs = {"learning_rate": 5e-4}

# train_rl_agent(
#     run_name=run_name,
#     env_id=env_id,
#     env_kwargs=env_kwargs,
#     algo_name=algo,
#     model_kwargs=model_kwargs,
#     train_steps=int(5e6),
#     resume=False,
# )

results, _ = load_eval_data(run_name)
results["timestep"] = results["timestep"] / 1e7
# results = results[results["timestep"] <= 1e4]

print(results)

results["return"] = (
    results.groupby("env")["return"].transform(lambda x: x.rolling(5).mean()) * 1.93
)

print(results)

fig, ax = plt.subplots()
sns.lineplot(data=results, x="timestep", y="return", ax=ax, color="#3B8B83")
ax.set(xlabel="Timestep $(\\times 10^5)$", ylabel="Mean eval return")
ax.set_title(
    f"Evaluation returns over training for DDPG in simple environment", fontsize=14
)
fig.tight_layout()
plt.show()


# # load trained policy
# policy = load_policy(os.path.join("../checkpoints", run_name, "best_model.zip"))
# enjoy_policy(policy, env_id, env_kwargs, seed=seed)
