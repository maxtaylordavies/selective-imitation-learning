from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import torch.utils.data as th_data
from imitation.data import types

from .transitions import MultiAgentTransitions
from .weighting import weight_agents_uniform
from ..types import WeightingFn


def make_data_loader(
    transitions: MultiAgentTransitions,
    batch_size: int,
    other_data: Dict[str, Any] = {},
    weight_fn: WeightingFn = weight_agents_uniform,
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
    agent_weights = weight_fn(transitions, other_data)
    print(f"Agent weights: {agent_weights}")
    sample_weights = agent_weights[transitions.agent_idxs]
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
