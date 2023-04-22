import numpy as np
import copy
import gymnasium as gym
from gymnasium import spaces

territories = [0,1,2,3,4,5]
adjacencies = [(0,1), (0,2), (1,3), (2,4), (3,5), (4,5)]
inverse_adjacencies = [(j,i) for (i,j) in adjacencies]
adjacencies = adjacencies + inverse_adjacencies

total_num_of_adjacencies = len(adjacencies)
movement_action_space = spaces.Box(low=0.0, high=1.0, shape=(total_num_of_adjacencies,))

MAX_INFANTRY_PER_TERRITORY = 50
player_infantry_shape = MAX_INFANTRY_PER_TERRITORY*np.ones(len(territories))
territory_owner_shape = np.ones(len(territories))

observation_space = spaces.Dict({
    'player1_infantry': spaces.MultiDiscrete(player_infantry_shape),
    'player2_infantry': spaces.MultiDiscrete(player_infantry_shape),
    'territory_owner': spaces.MultiDiscrete(territory_owner_shape),
})

opening_observation = {
    'player1_infantry': [10,2,2,0,0,0],
    'player2_infantry': [0,0,0,2,2,10],
    'territory_owner': [0,0,0,1,1,1]
}

class AxisAndAlliesGame(gym.Env):
    def __init__(self):
        pass

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # TODO: copy the opening observation
        self.observation = opening_observation
        self.current_player_turn = 0
        info = None
        return self.observation, info
    
    def step(self, action):
        terminated = False
        reward = 1
        info = None

        current_player_infantry = self.observation['player2_infantry'] if self.current_player_turn else self.observation['player1_infantry']
        infantry_to_move_per_adjacency = []
        contested_territories = []
        for percentage, (from_territory, to_territory) in zip(action, adjacencies):
            infantry_to_move = np.floor(percentage * current_player_infantry[from_territory])
            infantry_to_move_per_adjacency.append(infantry_to_move)

        for infantry_to_move, (from_territory, to_territory) in zip(infantry_to_move_per_adjacency, adjacencies):
            current_player_infantry[from_territory] -= infantry_to_move
            current_player_infantry[to_territory] += infantry_to_move
            if self.observation['territory_owner'][to_territory] != self.current_player_turn:
                contested_territories.append(to_territory)
        
        return self.observation, reward, terminated, False, info

game = AxisAndAlliesGame()
obs, info = game.reset()
print('first observation after reset: ', obs)

action = movement_action_space.sample()
print('random action: ', action)

observation, reward, terminated, truncated, info = game.step(action)
print('observation after step: ', observation)
