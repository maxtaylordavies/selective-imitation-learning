from gymnasium.envs.registration import register

register(
    id="FruitWorld-v0",
    entry_point="selective_imitation_learning.environments.fruitworld:FruitWorld",
    max_episode_steps=1000,
)

register(
    id="MultiAgentFruitWorld-v0",
    entry_point="selective_imitation_learning.environments.ma_fruitworld:MultiAgentFruitWorld",
    max_episode_steps=1000,
)

register(
    id="MultiAgentFruitWorldRiver-v0",
    entry_point="selective_imitation_learning.environments.ma_fruitworld_river:MultiAgentFruitWorldRiver",
    max_episode_steps=1000,
)
