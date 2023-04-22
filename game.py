import numpy as np
import gymnasium as gym
from gymnasium import spaces

territories = [0,1,2,3,4,5]
adjacencies = [(0,1), (0,2), (1,3), (2,4), (3,5), (4,5)]
inverse_adjacencies = [(j,i) for (i,j) in adjacencies]
adjacencies = adjacencies + inverse_adjacencies

num_of_territories = len(territories)
num_of_adjacencies = len(adjacencies)
movement_action_space = spaces.Box(low=0.0, high=1.0, shape=(num_of_adjacencies,))

MAX_INFANTRY_PER_TERRITORY = 50
player_infantry_shape = MAX_INFANTRY_PER_TERRITORY*np.ones(num_of_territories)

observation_space = spaces.Dict({
    'player1_infantry': spaces.MultiDiscrete(player_infantry_shape),
    'player2_infantry': spaces.MultiDiscrete(player_infantry_shape),
    'territory_owner': spaces.MultiBinary(num_of_territories),
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
        reward = 1
        info = None

        current_player_infantry = self.observation['player2_infantry'] if self.current_player_turn else self.observation['player1_infantry']
        infantry_to_move_per_adjacency = []
        sum_of_infantry_leaving_each_territory = np.zeros(num_of_territories)
        contested_territories = []
        for percentage, (from_territory, to_territory) in zip(action, adjacencies):
            infantry_to_move = np.floor(percentage * current_player_infantry[from_territory])
            infantry_to_move_per_adjacency.append(infantry_to_move)
            sum_of_infantry_leaving_each_territory[from_territory] += infantry_to_move

        for existing_infantry, leaving_infantry in zip(current_player_infantry, sum_of_infantry_leaving_each_territory):
            if leaving_infantry > existing_infantry:
                # invalid move! terminating game. maybe we don't have to terminate?
                return self.observation, -1000, True, False, info

        for infantry_to_move, (from_territory, to_territory) in zip(infantry_to_move_per_adjacency, adjacencies):
            current_player_infantry[from_territory] -= infantry_to_move
            current_player_infantry[to_territory] += infantry_to_move
            if self.observation['territory_owner'][to_territory] != self.current_player_turn:
                contested_territories.append(to_territory)
        
        self.current_player_turn = not self.current_player_turn
        return self.observation, reward, False, False, info

game = AxisAndAlliesGame()
obs, info = game.reset()
print('first observation after reset: ', obs)

num_of_steps = 2
for i in range(num_of_steps):
    action = movement_action_space.sample()
    print(f'random action {i+1}: ', action)

    observation, reward, terminated, truncated, info = game.step(action)
    if terminated or truncated:
        print('Game over..')
        break
    else:
        print(f'after step {i+1}: ', observation, reward, terminated)

