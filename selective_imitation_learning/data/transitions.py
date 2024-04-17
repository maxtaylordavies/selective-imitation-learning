import dataclasses
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import gymnasium
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from numpy.typing import NDArray
import torch.utils.data as th_data
from imitation.data import types, rollout
from imitation.policies.serialize import load_policy as load_expert_policy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm

from ..utils import load_policy


@dataclasses.dataclass(frozen=True)
class SplitMultiAgentTransitions(types.Transitions):
    # array of agent indices - agent_idxs[i] is the index of the agent associated with transition i
    agent_idxs: jnp.ndarray

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
        indices = jnp.where(self.agent_idxs == agent)[0]
        return self[indices]

    def sample_uniform(self, seed, n: int):
        assert n <= len(self)
        idxs = jr.permutation(jr.PRNGKey(seed), len(self))[:n]
        return self[idxs]


@dataclasses.dataclass(frozen=True)
class UnifiedMultiAgentTransitions(types.Transitions):
    def __getitem__(self, key):
        """See TransitionsMinimal docstring for indexing and slicing semantics."""
        d = types.dataclass_quick_asdict(self)
        d_item = {k: v[key] for k, v in d.items()}

        if not isinstance(key, int):
            return dataclasses.replace(self, **d_item)
        else:
            return d_item

    def sample_uniform(self, seed, n: int):
        assert n <= len(self)
        idxs = jr.permutation(jr.PRNGKey(seed), len(self))[:n]
        return self[idxs]


def generate_data(
    env: gymnasium.Env,
    seed: int,
    min_timesteps: int,
    policy_func: Callable,
    policy_func_kwargs: Mapping[str, Any] = {},
) -> Tuple[List, List, List, List, List]:
    obss, acts, next_obss, dones, infos = [], [], [], [], []
    timesteps, pbar = 0, tqdm(total=min_timesteps)
    while timesteps < min_timesteps:
        obs, _ = env.reset(seed=seed + timesteps)
        done = False
        while not done:
            obss.append(obs)

            act = policy_func(obs, **policy_func_kwargs)
            next_obs, _, done, _, info = env.step(act)

            acts.append(act)
            next_obss.append(next_obs)
            dones.append(done)
            infos.append(info)

            obs = next_obs
            timesteps += 1
            pbar.update(1)

    return (obss, acts, next_obss, dones, infos)


def generate_split_dataset(
    env: gymnasium.Env,
    seed: int,
    min_timesteps_per_agent: int,
    policy_funcs: Sequence[Callable],
    policy_func_kwargs: Sequence[Mapping[str, Any]] = [],
) -> SplitMultiAgentTransitions:
    assert len(policy_func_kwargs) == 0 or len(policy_func_kwargs) == len(policy_funcs)
    if len(policy_func_kwargs) == 0:
        policy_func_kwargs = [{}] * len(policy_funcs)

    # initalise arrays to store data
    obss, acts, next_obss, dones, infos, agent_idxs = [], [], [], [], [], []

    # sample transitions for each agent
    for m, (policy_func, kwargs) in enumerate(zip(policy_funcs, policy_func_kwargs)):
        data = generate_data(env, seed, min_timesteps_per_agent, policy_func, kwargs)
        obss += data[0]
        acts += data[1]
        next_obss += data[2]
        dones += data[3]
        infos += data[4]
        agent_idxs += [m] * len(data[0])

    # package into SplitMultiAgentTransitions object
    return SplitMultiAgentTransitions(
        obs=jnp.array(obss),
        acts=jnp.array(acts),
        next_obs=jnp.array(next_obss),
        infos=np.array(infos),
        dones=np.array(dones),
        agent_idxs=jnp.array(agent_idxs),
    )


def generate_unified_dataset(
    env: gymnasium.Env,
    seed: int,
    min_timesteps: int,
    policy_func: Callable,
    policy_func_kwargs: Mapping[str, Any] = {},
) -> UnifiedMultiAgentTransitions:
    data = generate_data(env, seed, min_timesteps, policy_func, policy_func_kwargs)
    return UnifiedMultiAgentTransitions(
        obs=jnp.array(data[0]),
        acts=jnp.array(data[1]),
        next_obs=jnp.array(data[2]),
        infos=np.array(data[4]),
        dones=np.array(data[3]),
    )


# def generate_demonstrations(
#     env: VecEnv,
#     rng: np.random.Generator,
#     agent_paths: List[str],
#     min_timesteps_per_agent: int = int(1e5),
#     algo_name="ppo",
# ) -> SplitMultiAgentTransitions:
#     rollouts, agent_idxs = [], []
#     for i, agent_path in enumerate(tqdm(agent_paths)):
#         expert = load_policy(agent_path)
#         _rollouts = rollout.rollout(
#             expert,
#             env,
#             rollout.make_sample_until(
#                 min_timesteps=min_timesteps_per_agent,
#             ),
#             rng=rng,
#             unwrap=False,
#         )
#         rollouts += _rollouts
#         agent_idxs += [i] * len(_rollouts)

#     return flatten_trajectories(rollouts, agent_idxs)


# def flatten_trajectories(
#     trajectories: List[types.Trajectory], agent_idxs: List[int]
# ) -> SplitMultiAgentTransitions:

#     def all_of_type(key, desired_type):
#         return all(
#             isinstance(getattr(traj, key), desired_type) for traj in trajectories
#         )

#     assert all_of_type("obs", np.ndarray)
#     assert all_of_type("acts", np.ndarray)
#     assert len(agent_idxs) == len(trajectories)

#     # mypy struggles without Any annotation here.
#     # The necessary constraints are enforced above.
#     keys = ["obs", "next_obs", "acts", "dones", "infos", "agent_idxs"]
#     parts: Mapping[str, List[Any]] = {key: [] for key in keys}
#     for i, traj in enumerate(trajectories):
#         parts["agent_idxs"].append([agent_idxs[i]] * len(traj.acts))
#         parts["acts"].append(traj.acts)

#         obs = traj.obs
#         parts["obs"].append(obs[:-1])
#         parts["next_obs"].append(obs[1:])

#         dones = np.zeros(len(traj.acts), dtype=bool)
#         dones[-1] = traj.terminal
#         parts["dones"].append(dones)

#         infos = traj.infos if traj.infos is not None else np.array([{}] * len(traj))
#         parts["infos"].append(infos)

#     def concat(key, item):
#         assert len(item) > 0
#         arr = np.concatenate(item)
#         return arr if key == "infos" else jnp.array(arr)

#     cat_parts = {key: concat(key, part_list) for key, part_list in parts.items()}
#     lengths = set(map(len, cat_parts.values()))
#     assert len(lengths) == 1, f"expected one length, got {lengths}"

#     return SplitMultiAgentTransitions(**cat_parts)
