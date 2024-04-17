import collections
import dataclasses
import functools
import itertools
import os
from struct import Struct
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm
from imitation.algorithms import base as algo_base
from imitation.data import types, rollout
from imitation.policies.serialize import load_policy
from imitation.policies import base as policy_base
from imitation.util import util
from stable_baselines3.common import policies, torch_layers, utils, vec_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.save_util import load_from_zip_file
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from ..data import (
    SplitMultiAgentTransitions,
    BCBatchLoader,
    generate_demonstrations,
    weight_agents_uniform,
    dataset_ll,
)
from ..utils import save_policy
from ..types import FeaturisingFn, WeightingFn
from .callback import EvalCallback


class ILEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, vec_env.VecEnv],
        batch_size: int,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
    ):
        super().__init__(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
        )
        self.batch_size = batch_size
        self.num_timesteps = 0
        self.last_eval = 0
        self.metrics = {
            **self.metrics,
            "demonstrator_weights": [],
        }

    def set_model(self, model):
        self.model = model

    def on_batch_end(self, model):
        self.set_model(model)
        self.num_timesteps += self.batch_size
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            self._run_eval()

    def save_demonstrator_weights(self, weights: jax.Array):
        self.metrics["demonstrator_weights"].append(weights.tolist())


@dataclasses.dataclass(frozen=True)
class BCTrainingMetrics:
    """Container for the different components of behavior cloning loss."""

    neglogp: torch.Tensor
    entropy: Optional[torch.Tensor]
    ent_loss: torch.Tensor  # set to 0 if entropy is None
    prob_true_act: torch.Tensor
    l2_norm: torch.Tensor
    l2_loss: torch.Tensor
    loss: torch.Tensor


@dataclasses.dataclass(frozen=True)
class BCLossCalculator:
    """Functor to compute the loss used in Behavior Cloning."""

    ent_weight: float
    l2_weight: float

    def __call__(
        self,
        policy: policies.ActorCriticPolicy,
        obs: Union[
            types.AnyTensor,
            types.DictObs,
            Dict[str, np.ndarray],
            Dict[str, torch.Tensor],
        ],
        acts: Union[torch.Tensor, np.ndarray],
    ) -> BCTrainingMetrics:
        """Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            policy: The actor-critic policy whose loss is being computed.
            obs: The observations seen by the expert.
            acts: The actions taken by the expert.

        Returns:
            A BCTrainingMetrics object with the loss and all the components it
            consists of.
        """
        tensor_obs = types.map_maybe_dict(
            util.safe_to_tensor,
            types.maybe_unwrap_dictobs(obs),
        )
        acts = util.safe_to_tensor(acts)

        # policy.evaluate_actions's type signatures are incorrect.
        # See https://github.com/DLR-RM/stable-baselines3/issues/1679
        (_, log_prob, entropy) = policy.evaluate_actions(
            tensor_obs,  # type: ignore[arg-type]
            acts,
        )
        prob_true_act = torch.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean() if entropy is not None else None

        l2_norms = [torch.sum(torch.square(w)) for w in policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
        # sum of list defaults to float(0) if len == 0.
        assert isinstance(l2_norm, torch.Tensor)

        ent_loss = -self.ent_weight * (
            entropy if entropy is not None else torch.zeros(1)
        )
        neglogp = -log_prob
        l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss + l2_loss

        return BCTrainingMetrics(
            neglogp=neglogp,
            entropy=entropy,
            ent_loss=ent_loss,
            prob_true_act=prob_true_act,
            l2_norm=l2_norm,
            l2_loss=l2_loss,
            loss=loss,
        )


@dataclasses.dataclass(frozen=True)
class OmegasLossCalculator:
    sample_fn: Callable[[], SplitMultiAgentTransitions]
    featurise_fn: FeaturisingFn

    def __call__(
        self,
        omegas: jax.Array,
    ):
        transitions = self.sample_fn()
        return -dataset_ll(transitions, self.featurise_fn, omegas)


class SelectiveBC(algo_base.DemonstrationAlgorithm):
    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        rng: np.random.Generator,
        batch_size: int = 64,
        demonstrations: SplitMultiAgentTransitions,
        featuriser: FeaturisingFn,
        weighting_fn: WeightingFn,
        omegas_self: jax.Array,
        omegas_update_freq: int = 10,
        omegas_batch_size: int = 128,
        omegas_update_coeff: float = 0.1,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Mapping[str, Any] = {},
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        device: Union[str, torch.device] = "auto",
    ):
        """Builds BC.

        Args:
            observation_space: the observation space of the environment.
            action_space: the action space of the environment.
            rng: the random state to use for the random number generator.
            policy: a Stable Baselines3 policy; if unspecified,
                defaults to `FeedForward32Policy`.
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            batch_size: The number of samples in each batch of expert data.
            minibatch_size: size of minibatch to calculate gradients over.
                The gradients are accumulated until `batch_size` examples
                are processed before making an optimization step. This
                is useful in GPU training to reduce memory usage, since
                fewer examples are loaded into memory at once,
                facilitating training with larger batch sizes, but is
                generally slower. Must be a factor of `batch_size`.
                Optional, defaults to `batch_size`.
            optimizer_cls: optimiser to use for supervised training.
            optimizer_kwargs: keyword arguments, excluding learning rate and
                weight decay, for optimiser construction.
            ent_weight: scaling applied to the policy's entropy regularization.
            l2_weight: scaling applied to the policy's L2 regularization.
            device: name/identity of device to place policy on.
            custom_logger: Where to log to; if None (default), creates a new logger.

        Raises:
            ValueError: If `weight_decay` is specified in `optimizer_kwargs` (use the
                parameter `l2_weight` instead), or if the batch size is not a multiple
                of the minibatch size.
        """
        self.batch_size = batch_size
        super().__init__(
            demonstrations=demonstrations,
        )
        self.action_space = action_space
        self.observation_space = observation_space
        self.weighting_fn = weighting_fn
        self.omegas_self = omegas_self
        self.omegas_update_freq = omegas_update_freq
        self.omegas_batch_size = omegas_batch_size
        self.omegas_update_coeff = omegas_update_coeff
        self.rng = rng

        extractor = (
            torch_layers.CombinedExtractor
            if isinstance(observation_space, gym.spaces.Dict)
            else torch_layers.FlattenExtractor
        )
        self._policy = policy_base.FeedForward32Policy(
            observation_space=observation_space,
            action_space=action_space,
            # Set lr_schedule to max value to force error if policy.optimizer
            # is used by mistake (should use self.policy_optim instead).
            lr_schedule=lambda _: torch.finfo(torch.float32).max,
            features_extractor_class=extractor,
        ).to(utils.get_device(device))
        assert self.policy.observation_space == self.observation_space
        assert self.policy.action_space == self.action_space

        self.policy_optim = optimizer_cls(
            self.policy.parameters(),
            **optimizer_kwargs,
        )
        self.bc_loss_calculator = BCLossCalculator(ent_weight, l2_weight)

        self.num_agents = len(jnp.unique(demonstrations.agent_idxs))
        self.num_features = featuriser(jnp.zeros(observation_space.shape)).shape[0]
        self.omegas = jr.normal(jr.PRNGKey(0), (self.num_agents, self.num_features))
        self.omegas_optim = optax.adam(1e-2)
        self.omegas_opt_state = self.omegas_optim.init(self.omegas)
        self.omegas_loss_calculator = OmegasLossCalculator(
            functools.partial(
                self.batch_loader.sample_batch, None, True, self.omegas_batch_size
            ),
            featuriser,
        )

    def set_demonstrations(self, demonstrations: SplitMultiAgentTransitions) -> None:
        self.batch_loader = BCBatchLoader(demonstrations, self.batch_size)

    @property
    def policy(self) -> policies.ActorCriticPolicy:
        return self._policy

    def initialise_demonstrator_weights(self):
        self.demonstrator_weights = jnp.ones(self.num_agents) / self.num_agents

    def update_demonstrator_weights(self, update_omegas: bool = True):
        if update_omegas:
            omegas_grad = jax.grad(self.omegas_loss_calculator)(self.omegas)
            omegas_updates, self.omegas_opt_state = self.omegas_optim.update(
                omegas_grad, self.omegas_opt_state, self.omegas
            )
            self.omegas = optax.apply_updates(self.omegas, omegas_updates)
        new_weights = self.weighting_fn(jnp.array(self.omegas), self.omegas_self)
        self.demonstrator_weights = (
            1 - self.omegas_update_coeff
        ) * self.demonstrator_weights + self.omegas_update_coeff * new_weights

    def train(
        self,
        *,
        n_batches: int = 1,
        callback: Optional[ILEvalCallback] = None,
    ):
        self.policy_optim.zero_grad()

        print(f"n_batches: {n_batches}")

        # set initial demonstrator weights
        self.initialise_demonstrator_weights()
        if callback is not None:
            callback.save_demonstrator_weights(self.demonstrator_weights)

        for batch_idx in tqdm(range(n_batches)):
            # sample batch
            batch: SplitMultiAgentTransitions = self.batch_loader.sample_batch(
                self.demonstrator_weights
            )

            # convert array to tensors
            obs, acts = np.array(batch.obs), np.array(batch.acts)
            obs_tensor: Union[torch.Tensor, Dict[str, torch.Tensor]]
            obs_tensor = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(x, device=self.policy.device),
                types.maybe_unwrap_dictobs(obs),
            )
            acts = util.safe_to_tensor(acts, device=self.policy.device)

            # compute BC loss and update policy
            training_metrics = self.bc_loss_calculator(self.policy, obs_tensor, acts)
            loss = training_metrics.loss
            loss.backward()
            self.policy_optim.step()
            self.policy_optim.zero_grad()

            # maybe update demonstrator weights
            if batch_idx % self.omegas_update_freq == 0:
                self.update_demonstrator_weights()

            if callback is not None:
                callback.save_demonstrator_weights(self.demonstrator_weights)
                callback.on_batch_end(self)


def train_bc_agent(
    rng: np.random.Generator,
    run_name: str,
    env_id: str,
    env_kwargs: Dict,
    demonstrations: SplitMultiAgentTransitions,
    featuriser: FeaturisingFn,
    omegas_self: jax.Array,
    weight_fn: WeightingFn = weight_agents_uniform,
    train_epochs: int = 1,
    batch_size: int = 64,
    omegas_batch_size: int = 5000,
    omegas_update_freq: int = 20,
    omegas_update_coeff: float = 0.1,
    eval_freq: int = 20000,
    n_eval_envs: int = 10,
    n_eval_eps: int = 50,
    eval_seed: int = 0,
    log_dir: str = "../checkpoints",
):
    # create run directory if it doesn't exist
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    env = make_vec_env(
        env_id, n_envs=n_eval_envs, seed=eval_seed, env_kwargs=env_kwargs
    )

    # compute number of batches
    n_batches = int(train_epochs * (len(demonstrations) / batch_size))

    bc = SelectiveBC(
        rng=rng,
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=demonstrations,
        batch_size=batch_size,
        featuriser=featuriser,
        weighting_fn=weight_fn,
        omegas_self=omegas_self,
        omegas_update_freq=omegas_update_freq,
        omegas_batch_size=omegas_batch_size,
    )

    callback = ILEvalCallback(
        eval_env=env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_eps,
        batch_size=bc.batch_size,
        log_path=run_dir,
    )

    # run initial eval
    callback.set_model(bc)
    callback._run_eval()

    # train behavior cloning model
    bc.train(n_batches=n_batches, callback=callback)

    # save final model
    save_policy(bc.policy, os.path.join(run_dir, "final_model"))
