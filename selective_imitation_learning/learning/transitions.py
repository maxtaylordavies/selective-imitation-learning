import dataclasses
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
from imitation.data import types, rollout
from imitation.policies.serialize import load_policy as load_expert_policy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.env_util import make_vec_env


@dataclasses.dataclass(frozen=True)
class MultiAgentTransitions(types.Transitions):
    # array of agent ids - agent_ids[i] is the id of the agent associated with transition i
    agent_ids: np.ndarray

    # dict that maps agent ids to arbitrary information about each agent
    agent_info: Dict[int, Any]

    def __post_init__(self):
        """Performs input validation: check shapes & dtypes match docstring."""
        super().__post_init__()
        if len(self.agent_ids) != len(self.obs):
            raise ValueError(
                "agent_ids should have the same length as obs: "
                f"{len(self.agent_ids)} != {len(self.obs)}",
            )

    def __getitem__(self, key):
        """See TransitionsMinimal docstring for indexing and slicing semantics."""
        d = types.dataclass_quick_asdict(self)
        d_item = {k: v[key] for k, v in d.items() if k != "agent_info"}

        if isinstance(key, slice):
            return dataclasses.replace(self, **d_item)
        else:
            assert isinstance(key, int)
            return d_item

    def get_agent_info(self, transition_index: int) -> Any:
        """Returns the agent info for the agent associated with the transition at index transition_index."""
        return self.agent_info[self.agent_ids[transition_index]]


def generate_demo_transitions(
    env: VecEnv,
    rng: np.random.Generator,
    agent_paths: List[str],
    agent_infos: List[Dict[str, Any]] = [],
    min_timesteps_per_agent: int = int(1e5),
    algo_name="ppo",
) -> MultiAgentTransitions:
    rollouts, agent_ids, agent_info = [], [], {}
    for i, agent_path in enumerate(agent_paths):
        expert = load_expert_policy(algo_name, env, path=agent_path)
        _rollouts = rollout.rollout(
            expert,
            env,
            rollout.make_sample_until(
                min_timesteps=min_timesteps_per_agent,
            ),
            rng=rng,
            unwrap=False,
        )
        rollouts += _rollouts
        agent_ids += [i] * len(_rollouts)
        agent_info[i] = agent_infos[i] if len(agent_infos) else {}

    return flatten_trajectories(rollouts, agent_ids, agent_info)


def flatten_trajectories(
    trajectories: List[types.Trajectory],
    agent_ids: List[int],
    agent_info: Dict[int, Any],
) -> MultiAgentTransitions:

    def all_of_type(key, desired_type):
        return all(
            isinstance(getattr(traj, key), desired_type) for traj in trajectories
        )

    assert all_of_type("obs", types.DictObs) or all_of_type("obs", np.ndarray)
    assert all_of_type("acts", np.ndarray)
    assert len(agent_ids) == len(trajectories)

    # mypy struggles without Any annotation here.
    # The necessary constraints are enforced above.
    keys = ["obs", "next_obs", "acts", "dones", "infos", "agent_ids"]
    parts: Mapping[str, List[Any]] = {key: [] for key in keys}
    for i, traj in enumerate(trajectories):
        parts["agent_ids"].append([agent_ids[i]] * len(traj.acts))
        parts["acts"].append(traj.acts)

        obs = traj.obs
        parts["obs"].append(obs[:-1])
        parts["next_obs"].append(obs[1:])

        dones = np.zeros(len(traj.acts), dtype=bool)
        dones[-1] = traj.terminal
        parts["dones"].append(dones)

        infos = traj.infos if traj.infos is not None else np.array([{}] * len(traj))
        parts["infos"].append(infos)

    cat_parts = {
        key: types.concatenate_maybe_dictobs(part_list)
        for key, part_list in parts.items()
    }
    lengths = set(map(len, cat_parts.values()))
    assert len(lengths) == 1, f"expected one length, got {lengths}"

    return MultiAgentTransitions(**cat_parts, agent_info=agent_info)
