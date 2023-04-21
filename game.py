import numpy as np

import gymnasium as gym
from gymnasium import spaces


class AxisAndAlliesGame(gym.Env):
    def __init__(self):
        pass

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        observation = None
        info = None
        return observation, info
    
    def step(self, action):
        terminated = False
        reward = 1
        observation = None
        info = None

        return observation, reward, terminated, False, info

game = AxisAndAlliesGame()
obs, info = game.reset()
print(obs, info)