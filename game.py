import numpy as np

import gymnasium as gym
from gymnasium import spaces

territories = [1,2,3,4,5,6]
adjacencies = [
    [2,3], # 1
    [1,4], # 2
    [1,5], # 3
    [2,6], # 4
    [3,6], # 5
    [4,5] # 6
]
total_num_of_adjacencies = np.array(adjacencies).size
movement_action_space = spaces.Box(low=0.0, high=1.0, shape=(total_num_of_adjacencies,))

MAX_INFANTRY_PER_TERRITORY = 50
discrete_shape = MAX_INFANTRY_PER_TERRITORY*np.ones((len(territories),2))
observation_space = spaces.MultiDiscrete(discrete_shape)

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
print('first observation after reset: ', obs)

action = movement_action_space.sample()
print('random action: ', action)

observation, reward, terminated, truncated, info = game.step(action)
print('observation after step: ', observation)

print('random observation: ', observation_space.sample())