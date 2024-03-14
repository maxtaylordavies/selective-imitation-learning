import gym
import numpy as np
import pygame


class FruitWorld(gym.Env):
    def __init__(
        self,
        grid_size: int,
        fruits_per_type: int,
        preferences: np.ndarray,
        render_mode=None,
    ):
        self.grid_size = grid_size
        self.fruits_per_type = fruits_per_type
        self.preferences = preferences
        self.num_fruits = len(self.preferences)
        self.render_mode = render_mode

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=self.num_fruits, shape=(grid_size, grid_size), dtype=np.int
        )
        self.action_space = gym.spaces.Discrete(4)

    def step(self, action):
        self._update_state(action)
        self.obs = self._get_obs()
        reward, done = self._compute_reward_and_done()
        return self.obs, reward, done, {}

    def reset(self, seed=None):
        # seed
        super().reset(seed=seed)

        # initialise state
        self.agent_pos = np.array([0, 0])
        self.fruits = {}
        for i in range(self.num_fruits):
            self.fruits[i] = np.random.randint(
                0, self.grid_size, size=(self.fruits_per_type, 2)
            )

        # return initial observation
        self.obs = self._get_obs()
        return self.obs, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size))
        obs[self.agent_pos[0], self.agent_pos[1]] = -1
        for i in range(self.num_fruits):
            for fruit in self.fruits[i]:
                obs[fruit[0], fruit[1]] = i
        return obs

    def _update_state(self, action):
        pass

    def _compute_reward_and_done(self):
        return 0, False

    def _render_frame(self):
        if self.render_mode == "human":
            if self.window is None:
                self.window = pygame.display.set_mode(
                    (self.grid_size * 50, self.grid_size * 50)
                )

            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.fill((255, 255, 255))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    pygame.draw.rect(
                        self.window,
                        (0, 0, 0),
                        (i * 50, j * 50, 50, 50),
                        1,
                    )
                    if self.obs[i, j] == -1:
                        pygame.draw.circle(
                            self.window,
                            (0, 0, 255),
                            (i * 50 + 25, j * 50 + 25),
                            20,
                        )
                    elif self.obs[i, j] >= 0:
                        pygame.draw.circle(
                            self.window,
                            (255, 0, 0),
                            (i * 50 + 25, j * 50 + 25),
                            20,
                        )
            pygame.display.flip()
            self.clock.tick(10)
