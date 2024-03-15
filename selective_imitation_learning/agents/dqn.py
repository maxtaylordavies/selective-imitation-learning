from copy import deepcopy
from itertools import accumulate
import os
from typing import Dict, Iterable, List, Optional

import gymnasium as gym
import numpy as np
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
import torch.nn.functional as F
from torch.optim import Adam

from .agent import Agent
from ..networks.fc_nework import FCNetwork
from ..replay import Transition


class DQN(Agent):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        max_timesteps: int,
        epsilon_decay_strategy: str = "constant",
        epsilon_decay: Optional[float] = None,
        exploration_fraction: Optional[float] = None,
        **kwargs,
    ):
        """The constructor of the DQN agent class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma
        :param epsilon_start (float): initial value of epsilon for epsilon-greedy action selection
        :param epsilon_min (float): minimum value of epsilon for epsilon-greedy action selection
        :param epsilon_decay (float, optional): decay rate of epsilon for epsilon-greedy action. If not specified,
                                                epsilon will be decayed linearly from epsilon_start to epsilon_min.
        """
        super().__init__(action_space, observation_space)

        obs_shape = self.observation_space.shape
        assert (
            obs_shape is not None and len(obs_shape) > 0
        ), "Observation space shape must be defined and non-empty"
        assert hasattr(self.action_space, "n"), "Action space must be discrete"

        # set up networks and optimisers
        self.critics_net = FCNetwork(
            (int(np.prod(obs_shape)), *hidden_size, action_space.n),
            output_activation=torch.nn.Identity(),  # no activation function
        )
        self.critics_target = deepcopy(self.critics_net)
        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3
        )

        # define any hyperparameters
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.max_timesteps = max_timesteps
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min

        self.epsilon_decay_strategy = epsilon_decay_strategy
        if epsilon_decay_strategy == "constant":
            assert (
                epsilon_decay is None
            ), "epsilon_decay should be None for epsilon_decay_strategy == 'constant'"
            assert (
                exploration_fraction is None
            ), "exploration_fraction should be None for epsilon_decay_strategy == 'constant'"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = None
        elif self.epsilon_decay_strategy == "linear":
            assert (
                epsilon_decay is None
            ), "epsilon_decay is only set for epsilon_decay_strategy='exponential'"
            assert (
                exploration_fraction is not None
            ), "exploration_fraction must be set for epsilon_decay_strategy='linear'"
            assert exploration_fraction > 0, "exploration_fraction must be positive"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = exploration_fraction
        elif self.epsilon_decay_strategy == "exponential":
            assert (
                epsilon_decay is not None
            ), "epsilon_decay must be set for epsilon_decay_strategy='exponential'"
            assert (
                exploration_fraction is None
            ), "exploration_fraction is only set for epsilon_decay_strategy='linear'"
            self.epsilon_exponential_decay_factor = epsilon_decay
            self.exploration_fraction = None
        else:
            raise ValueError(
                "epsilon_decay_strategy must be either 'linear' or 'exponential'"
            )

        # update saveables dict
        self.saveables.update(
            {
                "critics_net": self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim": self.critics_optim,
            }
        )

        # define loss function
        self.loss_fn = torch.nn.MSELoss()

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Update hyperparameters

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """

        def epsilon_linear_decay(timestep, max_timestep):
            return self.epsilon_start - (
                self.epsilon_start - self.epsilon_min
            ) * timestep / (self.exploration_fraction * max_timestep)

        def epsilon_exponential_decay(timestep, max_timestep):
            new_epsilon = self.epsilon * (
                self.epsilon_exponential_decay_factor ** (1 / max_timestep)
            )
            return max(new_epsilon, self.epsilon_min)

        if self.epsilon_decay_strategy == "constant":
            pass
        elif self.epsilon_decay_strategy == "linear":
            self.epsilon = epsilon_linear_decay(timestep, max_timestep)
        elif self.epsilon_decay_strategy == "exponential":
            self.epsilon = epsilon_exponential_decay(timestep, max_timestep)
        else:
            raise ValueError(
                "epsilon_decay_strategy must be either 'constant', 'linear' or 'exponential'"
            )

    def act(self, obs: np.ndarray, explore: bool):
        """Select an action

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """
        # if explore is True, select random action with probability self.epsilon
        r = np.random.random()
        if explore and (r < self.epsilon):
            return self.action_space.sample()

        # select greedy action
        probs = self.critics_net(torch.tensor(obs, dtype=torch.float32))
        return torch.argmax(probs).item()

    def compute_loss(self, batch: Transition) -> torch.Tensor:
        """Computes the DQN loss

        :param batch (Transition): batch vector from replay buffer
        :return (torch.Tensor): DQN loss
        """
        q = self.critics_net(batch.states).gather(1, batch.actions.long())
        with torch.no_grad():
            q_next = self.critics_target(batch.next_states).max(dim=1)[0].unsqueeze(1)
        q_target = batch.rewards + (self.gamma * (1 - batch.done) * q_next)
        return self.loss_fn(q, q_target)

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update DQN parameters

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        # clear gradients and update hyperparameters
        self.critics_optim.zero_grad()
        self.schedule_hyperparameters(self.update_counter, self.max_timesteps)

        # compute the DQN loss
        q_loss = self.compute_loss(batch)

        # update the network parameters
        q_loss.backward()
        self.critics_optim.step()

        # possibly update the target network
        if self.update_counter % self.target_update_freq == 0:
            self.critics_target.hard_update(self.critics_net)
        self.update_counter += 1

        return {"q_loss": q_loss.item()}
