from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
from gymnasium.utils import seeding
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from numpy.typing import NDArray
import pygame

from selective_imitation_learning.constants import ENV_CONSTANTS
from selective_imitation_learning.utils import manhattan_dist, to_simplex
from .utils import Position, GridActions, delta_x_actions, delta_y_actions

ObsType = NDArray[np.float32]
ActionType = int


class FruitWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    state_space: gym.Space
    _np_random: np.random.Generator
    fruit_probs: Dict[int, NDArray[np.float32]]
    agent_pos: Position

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
        self.num_fruit = num_fruit
        self.fruit_prefs = fruit_prefs
        self.fruit_types_deterministic = fruit_types_deterministic
        self.fruit_type_probs = fruit_type_probs
        self.fruit_loc_means = fruit_loc_means
        self.fruit_loc_stds = fruit_loc_stds
        self.num_fruit_types = len(fruit_prefs)
        self.base_fruit_reward = base_fruit_reward
        self.step_cost = step_cost
        self.num_lava = num_lava
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.window = None
        self.clock = None

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
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=3,
            shape=(grid_size, grid_size),
            dtype=np.float32,
        )
        self.state_space = self.observation_space
        self.action_space = gym.spaces.Discrete(len(GridActions))

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

        # initialise agent position
        # x, y = self._np_random.choice(self.grid_size, size=2, replace=True)
        # self.agent_pos = Position(x, y)
        self.agent_pos = self.agent_start_positions[
            self._np_random.integers(0, len(self.agent_start_positions))
        ]

        # # initialise lava positions
        # empty_cells = set(self.possible_locs) - {self.agent_pos}
        # self.lava = self._np_random.choice(
        #     list(empty_cells),
        #     size=(self.num_lava,),
        #     replace=False,
        # )
        #

        # initialise fruit positions
        self.fruit_map = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self._place_fruit(self.num_fruit)

        # initialise counters
        self.consumed_counts = [0] * len(self.fruit_prefs)

        # return initial observation
        return self._get_obs(), {}

    def step(self, action: ActionType) -> Tuple[ObsType, float, bool, bool, dict]:
        # move agent
        new_pos = self._get_new_pos(action)
        moved = not new_pos == self.agent_pos
        self.agent_pos = new_pos

        reward, done = -self.step_cost if moved else 0, False

        # check for lava-induced death or fruit consumption
        if moved:
            # if self.agent_pos in self.lava:
            #     done = True
            # else:
            pos_idx = self.agent_pos.to_np_idx()
            fruit = self.fruit_map[pos_idx] - 1
            if fruit >= 0:
                reward += float(self.fruit_prefs[fruit] * self.base_fruit_reward)
                self.consumed_counts[fruit] += 1
                self._place_fruit(
                    1, fruit_type=fruit if self.fruit_types_deterministic else None
                )
                # self.fruit_map = self.fruit_map.at[pos_idx].set(-1)
                self.fruit_map[pos_idx] = 0

        # check for termination
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            done = True

        info = {"consumed_counts": self.consumed_counts} if done else {}
        return self._get_obs(), reward, done, False, info

    def _get_new_pos(self, action: ActionType) -> Position:
        new_pos = Position(self.agent_pos.x, self.agent_pos.y)
        if action == GridActions.up:
            new_pos.y = max(0, new_pos.y - 1)
        elif action == GridActions.down:
            new_pos.y = min(self.grid_size - 1, new_pos.y + 1)
        elif action == GridActions.left:
            new_pos.x = max(0, new_pos.x - 1)
        elif action == GridActions.right:
            new_pos.x = min(self.grid_size - 1, new_pos.x + 1)
        return new_pos

    def _place_fruit(self, n, fruit_type=None) -> None:
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
            agent_r, agent_c = self.agent_pos.to_np_idx()
            probs[agent_r * self.grid_size + agent_c] = 0

            # sample locations
            idxs = self._np_random.choice(
                self.grid_size**2, size=count, p=probs / probs.sum(), replace=False
            )
            self.fruit_map[np.unravel_index(idxs, (self.grid_size, self.grid_size))] = (
                fruit_type + 1
            )

    def _get_obs(self, norm=False) -> ObsType:
        obs = self.fruit_map.copy().astype(np.float32)
        obs[self.agent_pos.to_np_idx()] = -1
        if not norm:
            return obs
        return (obs - obs.min()) / (obs.max() - obs.min())

    def _render_frame(self):
        if self.render_mode == "human":
            if self.window is None:
                self.window = pygame.display.set_mode(
                    (self.grid_size * 50, self.grid_size * 50)
                )

            if self.clock is None:
                self.clock = pygame.time.Clock()

            obs = self._get_obs(norm=False).T

            self.window.fill((255, 255, 255))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    pygame.draw.rect(
                        self.window,
                        (160, 160, 160),
                        (i * 50, j * 50, 50, 50),
                        2,
                    )
                    if obs[i, j] > 0 and obs[i, j] <= self.num_fruit_types:
                        pygame.draw.circle(
                            self.window,
                            ENV_CONSTANTS["fruit_colours"][int(obs[i, j] - 1)],
                            (i * 50 + 25, j * 50 + 25),
                            20,
                        )

            pygame.draw.rect(
                self.window,
                (0, 0, 0),
                (self.agent_pos.x * 50, self.agent_pos.y * 50, 50, 50),
                50,
            )

            pygame.display.flip()
            self.clock.tick(10)

    def render(self):
        self._render_frame()

    def close(self):
        pass


def featurise(obs: jnp.ndarray) -> jnp.ndarray:
    agent_pos = jnp.array(jnp.unravel_index(jnp.argmax(obs == 1.0), obs.shape))
    fruit_pos = jnp.array(
        [
            jnp.unravel_index(jnp.argmax(obs == (i + 1) * 0.25), obs.shape)
            for i in range(3)
        ]
    )
    feats = jnp.array([-manhattan_dist(agent_pos, f) for f in fruit_pos])
    return feats / (feats.sum() + 1e-6)


def expert_policy(obs: ObsType, fruit_prefs: NDArray) -> ActionType:
    target = np.argmax(fruit_prefs) + 1
    own_pos = np.array(np.unravel_index(np.argmax(obs == -1), obs.shape))
    target_pos = np.array(np.unravel_index(np.argmax(obs == target), obs.shape))
    delta_x, delta_y = np.sign(target_pos - own_pos)
    poss_actions = delta_x_actions[delta_x] + delta_y_actions[delta_y]
    return np.random.choice(poss_actions or list(GridActions))
