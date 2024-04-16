ENV_CONSTANTS = {
    "fruit_colours": [
        (239, 85, 59),
        (0, 204, 150),
        (99, 110, 250),
    ],
    "agent_colours": [
        (239, 85, 59),
        (0, 204, 150),
        (99, 110, 250),
    ],
}

PPO_DEFAULT_HYPERPARAMS = {
    "normalize_advantage": True,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "n_steps": 128,
    "batch_size": 64,
    "n_epochs": 10,
    "ent_coef": 0.0,
    "learning_rate": 2.5e-4,
    "clip_range": 0.2,
}

DQN_DEFAULT_HYPERPARAMS = {
    "learning_rate": 0.0000625,
    "batch_size": 64,
    "gamma": 0.99,
    "exploration_initial_eps": 0.9,
    "exploration_final_eps": 0.01,
    "target_update_interval": 80,
    "learning_starts": 10000,
    "train_freq": 4,
}
