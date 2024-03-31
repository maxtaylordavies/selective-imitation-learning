import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch.utils.data as th_data
from imitation.data import types, rollout
from imitation.policies.serialize import load_policy as load_expert_policy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm

from ..utils import load_policy


@dataclasses.dataclass(frozen=True)
class MultiAgentTransitions(types.Transitions):
    # array of agent indices - agent_idxs[i] is the index of the agent associated with transition i
    agent_idxs: np.ndarray

    def __post_init__(self):
        """Performs input validation: check shapes & dtypes match docstring."""
        super().__post_init__()
        if len(self.agent_idxs) != len(self.obs):
            raise ValueError(
                "agent_idxs should have the same length as obs: "
                f"{len(self.agent_idxs)} != {len(self.obs)}",
            )

    def __getitem__(self, key):
        """See TransitionsMinimal docstring for indexing and slicing semantics."""
        d = types.dataclass_quick_asdict(self)
        d_item = {k: v[key] for k, v in d.items()}

        if not isinstance(key, int):
            return dataclasses.replace(self, **d_item)
        else:
            return d_item

    def get_for_agent(self, agent: int):
        indices = np.where(self.agent_idxs == agent)[0]
        return self[indices]


def generate_demonstrations(
    env: VecEnv,
    rng: np.random.Generator,
    agent_paths: List[str],
    min_timesteps_per_agent: int = int(1e5),
    algo_name="ppo",
) -> MultiAgentTransitions:
    rollouts, agent_idxs = [], []
    for i, agent_path in enumerate(tqdm(agent_paths)):
        expert = load_policy(agent_path)
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
        agent_idxs += [i] * len(_rollouts)

    return flatten_trajectories(rollouts, agent_idxs)


def flatten_trajectories(
    trajectories: List[types.Trajectory], agent_idxs: List[int]
) -> MultiAgentTransitions:

    def all_of_type(key, desired_type):
        return all(
            isinstance(getattr(traj, key), desired_type) for traj in trajectories
        )

    assert all_of_type("obs", types.DictObs) or all_of_type("obs", np.ndarray)
    assert all_of_type("acts", np.ndarray)
    assert len(agent_idxs) == len(trajectories)

    # mypy struggles without Any annotation here.
    # The necessary constraints are enforced above.
    keys = ["obs", "next_obs", "acts", "dones", "infos", "agent_idxs"]
    parts: Mapping[str, List[Any]] = {key: [] for key in keys}
    for i, traj in enumerate(trajectories):
        parts["agent_idxs"].append([agent_idxs[i]] * len(traj.acts))
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

    return MultiAgentTransitions(**cat_parts)
