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

FRUIT_FEATURES_MAP = {
    0: [0, 0],  # red, circle
    1: [0, 1],  # red, square
    2: [1, 0],  # green, circle
    3: [1, 1],  # green, square
}


class MultiAgentFruitWorldRiver(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    state_space: gym.Space
    _np_random: np.random.Generator
    fruit_probs: Dict[int, NDArray[np.float32]]
    terrain_map: NDArray[np.int8]
    agent_positions: List[Position]

    def __init__(
        self,
        grid_size: int,
        expected_fruit_quants: List[int],
        agent_sides: List,
        fruit_sides: List,
        fruit_prefs: NDArray,
        fruit_type_probs: Optional[NDArray] = None,
        base_fruit_reward: float = 50.0,
        max_fruit_age: int = 30,
        step_cost: float = 0.5,
        max_steps: int = 100,
        render_mode=None,
    ):
        assert grid_size > 0, "Grid size must be positive"
        assert (
            len(expected_fruit_quants) > 0
        ), "Expected fruit quantities must be non-empty"
        assert max_steps > 0, "Maximum number of steps must be positive"
        assert max_fruit_age > 0, "Maximum fruit age must be positive"
        assert len(fruit_prefs) > 0, "Fruit preference vector must be non-empty"
        assert len(agent_sides) == len(
            fruit_prefs
        ), "Number of agents must match number of fruit preferences"
        assert (
            sum(expected_fruit_quants) < grid_size**2
        ), "Number of fruits cells exceeds grid size"

        self.grid_size = grid_size
        self.num_agents = fruit_prefs.shape[0]
        self.num_fruit_types = fruit_prefs.shape[1]
        self.expected_fruit_quants = expected_fruit_quants
        self.fruit_prefs = fruit_prefs
        self.fruit_type_probs = fruit_type_probs
        self.agent_sides = agent_sides
        self.fruit_sides = fruit_sides
        self.base_fruit_reward = base_fruit_reward
        self.max_fruit_age = max_fruit_age
        self.step_cost = step_cost
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None

        # default to uniform distribution over fruit types
        if self.fruit_type_probs is None:
            self.fruit_type_probs = np.ones(self.num_fruit_types) / self.num_fruit_types

        # compute the locations of all tiles south and north of the river respectively
        # (river runs along the diagonal from top left to bottom right)
        all_tiles = np.array(
            [(i, j) for i in range(grid_size) for j in range(grid_size)]
        )
        south_tiles = all_tiles[all_tiles[:, 0] < all_tiles[:, 1]]
        north_tiles = all_tiles[all_tiles[:, 0] > all_tiles[:, 1]]
        self.tile_locs = {"south": south_tiles, "north": north_tiles}

        self._init_fruit_distributions()

        # Define observation and action spaces
        #   - Observation space is a 2D grid with 3 channels -
        #       one for terrain types, one for agent
        #       positions and one for fruit positions
        #   - Action space is a MultiDiscrete space with each agent having
        #       5 possible actions (up, down, left, right, no-op)
        self.observation_space = self.state_space = gym.spaces.Box(
            low=0,
            high=max(self.num_agents, self.num_fruit_types),
            shape=(3, grid_size, grid_size),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete(
            len(GridActions) * np.ones(self.num_agents, dtype=int)
        )

        if self.render_mode:
            pygame.init()

    def _init_fruit_distributions(self):
        self.fruit_probs = {}
        for fruit_type, side in enumerate(self.fruit_sides):
            probs = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

            locs = (
                np.concatenate([self.tile_locs["south"], self.tile_locs["north"]])
                if side == "both"
                else self.tile_locs[side]
            )

            probs[locs[:, 1], locs[:, 0]] = 1
            self.fruit_probs[fruit_type] = probs / probs.sum()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        self._np_random, _ = seeding.np_random(seed)
        self.steps_taken = 0

        # init terrain map with water along the diagonal
        self.terrain_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for i in range(self.grid_size):
            self.terrain_map[i, i] = 1

        # initialise agent positions
        self._place_agents()

        # initialise fruit positions
        self.fruit_map = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.fruit_ages = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for ft, quant in enumerate(self.expected_fruit_quants):
            self._place_fruits(quant, fruit_type=ft)

        # initialise counters
        self.consumed_counts = np.zeros((self.num_agents, self.num_fruit_types))

        # return initial observation
        return self._get_obs(), {}

    def step(self, action: ActionType) -> Tuple[ObsType, NDArray, bool, bool, dict]:
        # move agent
        new_agent_positions = self._get_new_agent_positions(action)
        moveds = np.array(new_agent_positions) != np.array(self.agent_positions)
        self.agent_positions = new_agent_positions

        # determine rewards and remove any consumed fruit
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
                self._remove_fruit(pos_idx)

        # increment fruit ages, remove expired fruits
        self._age_fruits_and_remove_expired()

        # maybe generate new fruits
        self._maybe_generate_new_fruits()

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

        # do not allow agent to move into water
        if self.terrain_map[new_pos.y, new_pos.x] > 0:
            return cur_pos

        # do not allow agent to move into tile occupied by another agent
        if new_pos in others:
            return cur_pos

        return new_pos

    def _get_new_agent_positions(self, action: ActionType) -> List[Position]:
        new_positions = [Position.copy(pos) for pos in self.agent_positions]
        for m, pos in enumerate(new_positions):
            others = new_positions[:m] + new_positions[m + 1 :]
            new_positions[m] = self._move_agent(pos, others, action[m])
        return new_positions

    def _place_agents(self):
        aps, L = [], self.grid_size
        for m, side in enumerate(self.agent_sides):
            centre = np.array([L // 2, L // 2])
            delta = np.array([-L // 4, L // 4])
            pos = centre + (delta * (1 if side == "south" else -1))
            aps.append(Position(*pos))

            # idx = self._np_random.choice(len(self.tile_locs[side]))
            # aps.append(Position(*self.tile_locs[side][idx]))
        self.agent_positions = aps

    def _place_fruits(self, n, fruit_type=None):
        # first: if fruit type not specified, split n between fruit types
        if fruit_type is not None:
            fruit_counts = np.zeros(self.num_fruit_types, dtype=int)
            fruit_counts[fruit_type] = n
        else:
            assert self.fruit_type_probs is not None
            fruit_counts = self._np_random.multinomial(n, self.fruit_type_probs)

        # then sample locations for each fruit type
        for fruit_type, count in enumerate(fruit_counts):
            probs = self.fruit_probs[fruit_type].copy()

            # set probabilities of occupied locations to 0
            occ_rows, occ_cols = np.where(self.terrain_map + self.fruit_map > 0)
            probs[occ_rows, occ_cols] = 0
            for agent_pos in self.agent_positions:
                probs[agent_pos.to_np_idx()] = 0

            # sample locations
            idxs = self._np_random.choice(
                self.grid_size**2,
                size=count,
                p=probs.reshape(-1) / probs.sum(),
                replace=False,
            )
            locs = np.unravel_index(idxs, (self.grid_size, self.grid_size))

            # set fruit locations and ages
            self.fruit_map[locs] = fruit_type + 1
            self.fruit_ages[locs] = 1

    def _remove_fruit(self, loc):
        self.fruit_map[loc] = 0
        self.fruit_ages[loc] = 0

    def _age_fruits_and_remove_expired(self):
        # first, increment nonzero elements of fruit_ages
        self.fruit_ages[self.fruit_ages > 0] += 1

        # then, get indices of elements in fruit_ages > max_fruit_age
        rows, cols = np.where(self.fruit_ages > self.max_fruit_age)
        expired_locs = np.array([rows, cols]).T

        # each expired fruit has a 25% chance of being removed
        for loc in expired_locs:
            p = self._np_random.random()
            if p < 0.25:
                self._remove_fruit(loc)

    def _maybe_generate_new_fruits(self):
        base_p = 1 / self.max_fruit_age
        for fruit_type, n in enumerate(self.expected_fruit_quants):
            # generate new fruit with probability dependent on difference between
            # expected and actual number of fruits of this type currently on the grid
            delta = n - np.sum(self.fruit_map == fruit_type + 1)
            p = base_p + max(0, delta / n)
            if self._np_random.random() < p:
                self._place_fruits(1, fruit_type)

    def _get_obs(self, norm=False) -> ObsType:
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        # set terrain in first channel
        obs[0] = self.terrain_map.copy().astype(np.float32)

        # set agent positions in second channel
        for m, pos in enumerate(self.agent_positions):
            obs[1, pos.y, pos.x] = m + 1

        # set fruit positions in third channel
        obs[2] = self.fruit_map.copy().astype(np.float32)

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
        terrain_map = obs[0].T
        agent_map = obs[1].T
        fruit_map = obs[2].T

        self.window.fill((255, 255, 255))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pygame.draw.rect(
                    self.window,
                    (160, 160, 160),
                    (i * 50, j * 50, 50, 50),
                    2,
                )

                if terrain_map[i, j] > 0:
                    # draw water
                    pygame.draw.rect(
                        self.window,
                        (0, 0, 255),
                        (i * 50, j * 50, 50, 50),
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
                    fruit = int(fruit_map[i, j] - 1)
                    features = FRUIT_FEATURES_MAP[fruit]
                    color = ENV_CONSTANTS["fruit_colours"][features[0]]

                    if features[1] == 0:  # circle
                        pygame.draw.circle(
                            self.window,
                            color,
                            (i * 50 + 25, j * 50 + 25),
                            15,
                        )
                    else:  # square
                        pygame.draw.rect(
                            self.window,
                            color,
                            (i * 50 + 10, j * 50 + 10, 30, 30),
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
    terrain_map, agent_map, fruit_map = obs[0], obs[1], obs[2]

    def get_locs(_map, val):
        locs = np.where(_map == val)
        return np.array([locs[0], locs[1]]).T

    def choose_action(m):
        # compute current value of each fruit based on preferences and distance
        own_loc = get_locs(agent_map, m + 1)[0]

        # determine which side of the (diagonal) river the agent is on
        own_side = int(own_loc[0] < own_loc[1])

        # get locations of all fruits on same side of river as agent
        fruit_locs, fruit_types = [], []
        for f in range(len(fruit_prefs[m])):
            locs = get_locs(fruit_map, f + 1)
            locs = [loc for loc in locs if int(loc[0] < loc[1]) == own_side]
            fruit_locs += locs
            fruit_types += [f] * len(locs)
        fruit_locs, fruit_types = np.array(fruit_locs), np.array(fruit_types)

        # if no fruits on this side of the river, do nothing
        if len(fruit_locs) == 0:
            return GridActions.no_op
            # return np.random.choice(list(GridActions))

        # compute value of each available fruit based on preferences and distance
        dists = np.sum(np.abs(fruit_locs - own_loc), axis=1)
        vals = fruit_prefs[m, fruit_types] - c * dists

        # if all values negative, do nothing
        if np.all(vals < 0):
            return GridActions.no_op

        # pick a fruit to move towards using a boltzmann dist
        probs = boltzmann_np(vals, beta)
        target = np.random.choice(len(fruit_locs), p=probs)

        # then pick any action that moves towards the target fruit
        delta_x, delta_y = np.sign(fruit_locs[target] - own_loc)
        poss_actions = delta_x_actions[delta_x] + delta_y_actions[delta_y]
        return np.random.choice(poss_actions or list(GridActions))

    return np.array([choose_action(m) for m in range(M)])
