from dataclasses import dataclass
from typing import Optional, Tuple, Union

import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
import pygame

from selective_imitation_learning.constants import ENV_CONSTANTS

ObsType = np.ndarray
ActionType = int


@dataclass
class Position:
    x: int
    y: int

    def __eq__(self, __value: object) -> bool:
        return self.x == __value.x and self.y == __value.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


possible_fruit_locs = []
for i in [1, 3, 5]:
    for j in [1, 3, 5]:
        possible_fruit_locs.append(Position(i, j))


class FruitWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int,
        fruits_per_type: int,
        preferences: np.ndarray,
        max_steps: int = 10,
        render_mode=None,
    ):
        assert grid_size > 0, "Grid size must be positive"
        assert fruits_per_type > 0, "Number of fruits per type must be positive"
        assert np.isclose(
            np.sum(preferences), 1.0
        ), "Fruit preference vector must sum to 1"

        self.grid_size = grid_size
        self.fruits_per_type = fruits_per_type
        self.preferences = preferences
        self.max_steps = max_steps
        self.render_mode = render_mode

        """
        if human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. they will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=len(self.preferences),
            shape=(grid_size, grid_size),
            dtype=np.int64,
        )
        self.action_space = gym.spaces.Discrete(4)

    def step(self, action: ActionType) -> Tuple[ObsType, float, bool, bool, dict]:
        print(self.fruits)
        for i in self.fruits:
            assert len(self.fruits[i]) == self.fruits_per_type

        # move agent
        new_pos = self._get_new_pos(action)
        moved = not np.all(new_pos == self.agent_pos)
        self.agent_pos = new_pos

        reward, done = -0.1 if moved else 0, False

        # check for fruit consumption
        if moved:
            for i in self.fruits:
                for j, fruit_pos in enumerate(self.fruits[i]):
                    if self.agent_pos == fruit_pos:
                        reward += self.preferences[i] * 20  # eat fruit
                        self.fruits[i][j] = self._random_pos()  # regenerate fruit
                        break

        # check for termination
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            done = True

        self.obs = self._get_obs()
        return self.obs, reward, done, False, {}

    def _get_new_pos(self, action: ActionType) -> Position:
        new_pos = Position(self.agent_pos.x, self.agent_pos.y)
        if action == 0:  # up
            new_pos.y = max(0, new_pos.y - 1)
        elif action == 1:  # down
            new_pos.y = min(self.grid_size - 1, new_pos.y + 1)
        elif action == 2:  # left
            new_pos.x = max(0, new_pos.x - 1)
        elif action == 3:  # right
            new_pos.x = min(self.grid_size - 1, new_pos.x + 1)
        return new_pos

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        # seed
        if seed:
            self._np_random, _ = seeding.np_random(seed)
        else:
            self._np_random, _ = seeding.np_random()

        # initialise state
        self.steps_taken = 0
        self.agent_pos = Position(3, 3)
        initial_fruit_positions = self.np_random.choice(
            possible_fruit_locs,
            size=(len(self.preferences), self.fruits_per_type),
            replace=False,
        )
        self.fruits = {
            i: initial_fruit_positions[i] for i in range(len(self.preferences))
        }

        # return initial observation
        self.obs = self._get_obs()
        return self.obs, {}

    def render(self):
        self._render_frame()

    def close(self):
        pass

    def _random_pos(self) -> Position:
        all_locs = set(possible_fruit_locs.copy()) - {self.agent_pos}
        filled_locs = set([pos for fruit in self.fruits for pos in self.fruits[fruit]])
        empty_locs = list(all_locs - filled_locs)
        return empty_locs[self._np_random.integers(0, len(empty_locs))]

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.int64)
        obs[self.agent_pos.y, self.agent_pos.x] = -1
        for i in self.fruits:
            for fruit_pos in self.fruits[i]:
                obs[fruit_pos.y, fruit_pos.x] = i + 1
        return obs

    def _render_frame(self):
        if self.render_mode == "human":
            if self.window is None:
                self.window = pygame.display.set_mode(
                    (self.grid_size * 50, self.grid_size * 50)
                )

            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.fill((255, 255, 255))
            obs = self.obs.T
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    pygame.draw.rect(
                        self.window,
                        (160, 160, 160),
                        (i * 50, j * 50, 50, 50),
                        2,
                    )
                    if obs[i, j] == -1:
                        pygame.draw.rect(
                            self.window,
                            (0, 0, 0),
                            (i * 50, j * 50, 50, 50),
                            50,
                        )
                    elif obs[i, j] > 0:
                        pygame.draw.circle(
                            self.window,
                            ENV_CONSTANTS["fruit_colours"][obs[i, j] - 1],
                            (i * 50 + 25, j * 50 + 25),
                            20,
                        )
            pygame.display.flip()
            self.clock.tick(10)
