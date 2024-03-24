import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch.utils.data as th_data
from imitation.data import types, rollout
from imitation.policies.serialize import load_policy as load_expert_policy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.env_util import make_vec_env


@dataclasses.dataclass(frozen=True)
class MultiAgentTransitions(types.Transitions):
    # array of agent indices - agent_idxs[i] is the index of the agent associated with transition i
    agent_idxs: np.ndarray

    # list of info/metadata objects for each agent
    agent_infos: List[Dict]

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
        d_item = {k: v[key] for k, v in d.items() if k != "agent_infos"}

        if isinstance(key, slice):
            return dataclasses.replace(self, **d_item)
        else:
            assert isinstance(key, int)
            return d_item


def generate_demonstrations(
    env: VecEnv,
    rng: np.random.Generator,
    agent_paths: List[str],
    agent_infos: List[Dict] = [],
    min_timesteps_per_agent: int = int(1e5),
    algo_name="ppo",
) -> MultiAgentTransitions:
    rollouts, agent_idxs = [], []
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
        agent_idxs += [i] * len(_rollouts)

    return flatten_trajectories(rollouts, agent_idxs, agent_infos)


def flatten_trajectories(
    trajectories: List[types.Trajectory], agent_idxs: List[int], agent_infos: List[dict]
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

    return MultiAgentTransitions(**cat_parts, agent_infos=agent_infos)


def make_data_loader(
    transitions: MultiAgentTransitions,
    batch_size: int,
    weight_fn: Optional[Callable] = None,
    data_loader_kwargs: Optional[Mapping[str, Any]] = None,
) -> Iterable[types.TransitionMapping]:
    """Converts demonstration data to Torch data loader.

    Args:
        transitions: Transitions expressed directly as a `types.TransitionsMinimal`
            object, a sequence of trajectories, or an iterable of transition
            batches (mappings from keywords to arrays containing observations, etc).
        batch_size: The size of the batch to create. Does not change the batch size
            if `transitions` is already an iterable of transition batches.
        data_loader_kwargs: Arguments to pass to `th_data.DataLoader`.

    Returns:
        An iterable of transition batches.

    Raises:
        ValueError: if `transitions` is an iterable over transition batches with batch
            size not equal to `batch_size`; or if `transitions` is transitions or a
            sequence of trajectories with total timesteps less than `batch_size`.
        TypeError: if `transitions` is an unsupported type.
    """
    assert batch_size > 0, f"Batch size must be positive, got {batch_size}."
    assert (
        len(transitions) >= batch_size
    ), "Number of provided transitions cannot be smaller than batch size."

    # define sample weights for WeightedRandomSampler
    if weight_fn is None or len(transitions.agent_infos) == 0:
        sample_weights = np.ones(len(transitions.agent_idxs))
    else:
        agent_weights = weight_fn(transitions.agent_infos).astype(np.float32)
        sample_weights = agent_weights[transitions.agent_idxs]
    print(type(sample_weights), type(sample_weights[0]))
    sample_weights /= sample_weights.sum()

    sampler = th_data.WeightedRandomSampler(sample_weights, len(sample_weights))

    kwargs: Mapping[str, Any] = {
        "drop_last": True,
        **(data_loader_kwargs or {}),
    }

    return th_data.DataLoader(
        transitions,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=types.transitions_collate_fn,
        **kwargs,
    )
