ENV_CONSTANTS = {
    "fruit_colours": [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]
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
