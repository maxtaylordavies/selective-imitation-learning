from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium.utils import seeding
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from numpy.typing import NDArray
import pygame
import pygame.freetype

from selective_imitation_learning.constants import ENV_CONSTANTS
from selective_imitation_learning.utils import manhattan_dist, to_simplex, boltzmann_np
from .utils import Position, GridActions, delta_x_actions, delta_y_actions

ObsType = NDArray[np.float32]
ActionType = NDArray[np.int8]


class MultiAgentFruitWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    state_space: gym.Space
    _np_random: np.random.Generator
    fruit_probs: Dict[int, NDArray[np.float32]]
    agent_positions: List[Position]

    def __init__(
        self,
        grid_size: int,
        num_fruit: int,
        fruit_prefs: NDArray,
        fruit_types_deterministic: bool = True,
        fruit_type_probs: Optional[NDArray] = None,
        fruit_loc_means: Optional[NDArray] = None,
        fruit_loc_stds: Optional[NDArray] = None,
        base_fruit_reward: float = 10.0,
        step_cost: float = 0.5,
        num_lava: int = 0,
        max_steps: int = 10,
        render_mode=None,
    ):
        assert grid_size > 0, "Grid size must be positive"
        assert num_fruit > 0, "Number of fruits per type must be positive"
        assert num_lava >= 0, "Number of lava cells must be non-negative"
        assert max_steps > 0, "Maximum number of steps must be positive"
        assert len(fruit_prefs) > 0, "Fruit preference vector must be non-empty"
        assert (
            num_fruit + num_lava < grid_size**2
        ), "Number of fruits and lava cells exceeds grid size"

        self.grid_size = grid_size
        self.num_agents = fruit_prefs.shape[0]
        self.num_fruit_types = fruit_prefs.shape[1]
        self.num_fruit = num_fruit
        self.fruit_prefs = fruit_prefs
        self.fruit_types_deterministic = fruit_types_deterministic
        self.fruit_type_probs = fruit_type_probs
        self.fruit_loc_means = fruit_loc_means
        self.fruit_loc_stds = fruit_loc_stds
        self.base_fruit_reward = base_fruit_reward
        self.step_cost = step_cost
        self.num_lava = num_lava
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None

        self._init_fruit_distributions()

        self.agent_start_positions = np.array(
            [
                Position(1, 1),
                Position(1, self.grid_size - 2),
                Position(self.grid_size - 2, 1),
                Position(self.grid_size - 2, self.grid_size - 2),
                Position(self.grid_size // 2, self.grid_size // 2),
            ]
        )

        # Define observation and action spaces
        #   - Observation space is a 2D grid with 2 channels - one for agent
        #   positions and one for fruit positions
        #   - Action space is a MultiDiscrete space with each agent having
        #   5 possible actions (up, down, left, right, no-op)
        self.observation_space = self.state_space = gym.spaces.Box(
            low=0,
            high=max(self.num_agents, self.num_fruit_types),
            shape=(2, grid_size, grid_size),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete(
            len(GridActions) * np.ones(self.num_agents, dtype=int)
        )

        if self.render_mode:
            pygame.init()

    def _init_fruit_distributions(self):
        self.fruit_probs = {}

        row_idxs, col_idxs = np.indices((self.grid_size, self.grid_size))
        coords = np.column_stack((row_idxs.ravel(), col_idxs.ravel()))

        if not self.fruit_types_deterministic and self.fruit_type_probs is None:
            self.fruit_type_probs = np.ones(self.num_fruit_types) / self.num_fruit_types

        if self.fruit_loc_means is None:
            self.fruit_loc_means = np.ones((self.num_fruit_types, 2))
            self.fruit_loc_means *= self.grid_size // 2

        if self.fruit_loc_stds is None:
            self.fruit_loc_stds = np.ones(self.num_fruit_types)
            self.fruit_loc_stds *= self.grid_size // 4

        for i in range(self.num_fruit_types):
            sq_dists = np.sum((coords - self.fruit_loc_means[i]) ** 2, axis=1)
            self.fruit_probs[i] = np.exp(-sq_dists / (2 * self.fruit_loc_stds[i] ** 2))
            self.fruit_probs[i] /= self.fruit_probs[i].sum()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        self._np_random, _ = seeding.np_random(seed)
        self.steps_taken = 0

        # initialise agent positions
        self.agent_positions = self._np_random.choice(
            self.agent_start_positions,
            size=self.num_agents,
            replace=False,
        ).tolist()

        # initialise fruit positions
        self.fruit_map = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self._place_fruit(self.num_fruit)

        # initialise counters
        self.consumed_counts = np.zeros((self.num_agents, self.num_fruit_types))

        # return initial observation
        return self._get_obs(), {}

    def step(self, action: ActionType) -> Tuple[ObsType, NDArray, bool, bool, dict]:
        # move agent
        new_agent_positions = self._get_new_agent_positions(action)
        moveds = np.array(new_agent_positions) != np.array(self.agent_positions)
        self.agent_positions = new_agent_positions

        # determine rewards and regenerate any consumed fruit
        rewards, done = np.zeros(self.num_agents), False
        for m, (pos, moved) in enumerate(zip(self.agent_positions, moveds)):
            if not moved:
                continue
            rewards[m] -= self.step_cost
            pos_idx = pos.to_np_idx()
            fruit = self.fruit_map[pos_idx] - 1
            if fruit >= 0:
                rewards[m] += float(self.fruit_prefs[m, fruit] * self.base_fruit_reward)
                self.consumed_counts[m, fruit] += 1
                self._place_fruit(
                    1, fruit_type=fruit if self.fruit_types_deterministic else None
                )
                self.fruit_map[pos_idx] = 0

        # check for termination
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            done = True

        info = {"consumed_counts": self.consumed_counts} if done else {}
        return self._get_obs(), rewards, done, False, info

    def _move_agent(self, cur_pos, others, action: int) -> Position:
        new_pos = Position(cur_pos.x, cur_pos.y)
        if action == GridActions.up:
            new_pos.y = max(0, new_pos.y - 1)
        elif action == GridActions.down:
            new_pos.y = min(self.grid_size - 1, new_pos.y + 1)
        elif action == GridActions.left:
            new_pos.x = max(0, new_pos.x - 1)
        elif action == GridActions.right:
            new_pos.x = min(self.grid_size - 1, new_pos.x + 1)

        if new_pos in others:
            return cur_pos
        return new_pos

    def _get_new_agent_positions(self, action: ActionType) -> List[Position]:
        new_positions = [Position.copy(pos) for pos in self.agent_positions]
        for m, pos in enumerate(new_positions):
            others = new_positions[:m] + new_positions[m + 1 :]
            new_positions[m] = self._move_agent(pos, others, action[m])
        return new_positions

    def _place_fruit(self, n, fruit_type=None):
        # first split n between fruit types
        if fruit_type is not None:
            fruit_counts = np.zeros(self.num_fruit_types, dtype=int)
            fruit_counts[fruit_type] = n
        elif self.fruit_types_deterministic:
            assert n % self.num_fruit_types == 0
            fruit_counts = np.full(
                self.num_fruit_types, n // self.num_fruit_types, dtype=int
            )
        else:
            assert self.fruit_type_probs is not None
            fruit_counts = self._np_random.multinomial(n, self.fruit_type_probs)

        # then sample locations for each fruit type
        for fruit_type, count in enumerate(fruit_counts):
            # get distribution for this fruit type
            probs = self.fruit_probs[fruit_type].copy()

            # set probabilities of occupied locations to 0
            occ_rows, occ_cols = np.where(self.fruit_map > 0)
            probs[occ_rows * self.grid_size + occ_cols] = 0
            for agent_pos in self.agent_positions:
                r, c = agent_pos.to_np_idx()
                probs[r * self.grid_size + c] = 0

            # sample locations
            idxs = self._np_random.choice(
                self.grid_size**2, size=count, p=probs / probs.sum(), replace=False
            )
            self.fruit_map[np.unravel_index(idxs, (self.grid_size, self.grid_size))] = (
                fruit_type + 1
            )

    def _get_obs(self, norm=False) -> ObsType:
        obs = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)

        # set agent positions in first channel
        for m, pos in enumerate(self.agent_positions):
            obs[0, pos.y, pos.x] = m + 1

        # set fruit positions in second channel
        obs[1] = self.fruit_map.copy().astype(np.float32)

        if not norm:
            return obs
        return (obs - obs.min()) / (obs.max() - obs.min())

    def _render_frame(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            self.window = pygame.display.set_mode(
                (self.grid_size * 50, self.grid_size * 50)
            )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.font is None:
            self.font = pygame.freetype.SysFont("Helvetica", 24, bold=True)

        obs = self._get_obs(norm=False)
        agent_map = obs[0].T
        fruit_map = obs[1].T

        self.window.fill((255, 255, 255))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pygame.draw.rect(
                    self.window,
                    (160, 160, 160),
                    (i * 50, j * 50, 50, 50),
                    2,
                )
                if agent_map[i, j] > 0:
                    # draw square for agent
                    pygame.draw.rect(
                        self.window,
                        (66, 66, 66),
                        (i * 50, j * 50, 50, 50),
                    )
                    # draw agent number in middle of square
                    # self.font is set already
                    self.font.render_to(
                        self.window,
                        (i * 50 + 20, j * 50 + 20),
                        str(int(agent_map[i, j])),
                        (255, 255, 255),
                    )

                elif fruit_map[i, j] > 0:
                    pygame.draw.circle(
                        self.window,
                        ENV_CONSTANTS["fruit_colours"][int(fruit_map[i, j] - 1)],
                        (i * 50 + 25, j * 50 + 25),
                        20,
                    )

        pygame.display.flip()
        self.clock.tick(10)
        return self.window.copy()

    def render(self):
        return self._render_frame()

    def close(self):
        pass


def expert_policy(obs: ObsType, fruit_prefs: NDArray, beta=0.5, c=0.1) -> ActionType:
    M = len(fruit_prefs)  # number of agents
    agent_map, fruit_map = obs[0], obs[1]

    def get_loc(_map, val):
        return np.array(np.unravel_index(np.argmax(_map == val), _map.shape))

    def choose_action(m):
        # compute current value of each fruit based on preferences and distance
        own_loc = get_loc(agent_map, m + 1)
        fruit_locs = np.array(
            [get_loc(fruit_map, f + 1) for f in range(len(fruit_prefs[m]))]
        )
        dists = np.sum(np.abs(fruit_locs - own_loc), axis=1)
        vals = fruit_prefs[m] - c * dists

        # pick a fruit to move towards using a boltzmann dist
        probs = boltzmann_np(vals, beta)
        target = np.random.choice(len(fruit_prefs[m]), p=probs)

        # pick any action that moves towards the target fruit
        delta_x, delta_y = np.sign(fruit_locs[target] - own_loc)
        poss_actions = delta_x_actions[delta_x] + delta_y_actions[delta_y]
        return np.random.choice(poss_actions or list(GridActions))

    return np.array([choose_action(m) for m in range(M)])
