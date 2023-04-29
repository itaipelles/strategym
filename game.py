import numpy as np
import gymnasium as gym
from gymnasium import spaces
from battle_calculator import infantry_battle
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import copy
import time

territories = [0,1,2,3,4,5]
territory_values = [6,3,3,3,3,6]
adjacencies = [(0,1), (0,2), (1,3), (2,4), (3,5), (4,5)]
inverse_adjacencies = [(j,i) for (i,j) in adjacencies]
adjacencies = adjacencies + inverse_adjacencies

num_of_territories = len(territories)
num_of_adjacencies = len(adjacencies)

MAX_INFANTRY_PER_TERRITORY = 50
player_infantry_shape = MAX_INFANTRY_PER_TERRITORY*np.ones(num_of_territories)

P1_CAPITAL = 0
P2_CAPITAL = 5
capital_per_player = [P1_CAPITAL,P2_CAPITAL]

COST_OF_INFANTRY = 3

test_space = spaces.MultiDiscrete(player_infantry_shape)
test_space2 = spaces.MultiBinary(num_of_territories)

class AxisAndAlliesEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    observation_space = spaces.Dict({
        'player1_infantry': spaces.MultiDiscrete(player_infantry_shape),
        'player2_infantry': spaces.MultiDiscrete(player_infantry_shape),
        'territory_owner': spaces.MultiBinary(num_of_territories),
        })
    action_space : spaces.Box = spaces.Box(low=0.0, high=1.0, shape=(num_of_adjacencies,))
    opening_observation = {
        'player1_infantry': [10,2,2,0,0,0],
        'player2_infantry': [0,0,0,2,2,10],
        'territory_owner': [0,0,0,1,1,1]
        }
    
    def __init__(self,  render_mode="human"):
        self.G = nx.Graph()
        self.G.add_nodes_from(territories)
        self.G.add_edges_from(inverse_adjacencies)
        self.G_node_pos = nx.spring_layout(self.G) # sometimes this auto layout fails (see failed_layout_example.png). any solutions?
        self.render_mode = render_mode
        pass

    def reset(self, seed=None, options=None):
        # TODO: copy the opening observation
        self.observation = copy.deepcopy(self.opening_observation)
        self.current_player_turn = 0
        info = {}
        return self.observation, info
    
    def step(self, action):
        info = {}
        terminated = False
        truncated = False

        current_player_infantry = self.observation['player2_infantry'] if self.current_player_turn else self.observation['player1_infantry']
        other_player_infantry = self.observation['player1_infantry'] if self.current_player_turn else self.observation['player2_infantry']
        infantry_to_move_per_adjacency = []
        sum_of_infantry_leaving_each_territory = np.zeros(num_of_territories)
        contested_territories = []
        for percentage, (from_territory, to_territory) in zip(action, adjacencies):
            infantry_to_move = np.floor(percentage * current_player_infantry[from_territory])
            infantry_to_move_per_adjacency.append(infantry_to_move)
            sum_of_infantry_leaving_each_territory[from_territory] += infantry_to_move

        for existing_infantry, leaving_infantry in zip(current_player_infantry, sum_of_infantry_leaving_each_territory):
            if leaving_infantry > existing_infantry:
                # invalid move! skipping move. maybe we have to terminate?
                reward = -1000 * (leaving_infantry - existing_infantry)
                return self.observation, reward, terminated, truncated, info

        for infantry_to_move, (from_territory, to_territory) in zip(infantry_to_move_per_adjacency, adjacencies):
            current_player_infantry[from_territory] -= infantry_to_move
            current_player_infantry[to_territory] += infantry_to_move
            if current_player_infantry[to_territory]>0 and self.observation['territory_owner'][to_territory] != self.current_player_turn:
                contested_territories.append(to_territory)
        
        for territory in np.unique(contested_territories):
            new_attack_inf, new_defend_inf = infantry_battle(current_player_infantry[territory], other_player_infantry[territory])
            # print(f'Combat! territory {territory} attack {current_player_infantry[territory]} -> {new_attack_inf}, defend {other_player_infantry[territory]} -> {new_defend_inf}')
            current_player_infantry[territory] = new_attack_inf
            other_player_infantry[territory] = new_defend_inf
            if new_attack_inf > 0:
                self.observation['territory_owner'][territory] = self.current_player_turn

        p1_lost = self.observation['territory_owner'][P1_CAPITAL] == 1
        p2_lost = self.observation['territory_owner'][P2_CAPITAL] == 0
        if p1_lost or p2_lost:
            terminated = True

        if not terminated:
            # TODO: keep income value in observation from last turn and use that instead
            income = np.sum([territory_value for territory, territory_value in enumerate(territory_values) if self.observation['territory_owner'][territory] == self.current_player_turn])
            new_infantry_amount = np.floor(income/COST_OF_INFANTRY)
            current_player_infantry[capital_per_player[self.current_player_turn]] += new_infantry_amount
            current_player_units_value = COST_OF_INFANTRY * np.sum(current_player_infantry)
            other_player_units_value = COST_OF_INFANTRY * np.sum(other_player_infantry)
            reward = income + current_player_units_value - other_player_units_value
        else:
            reward = 1000

        self.current_player_turn = not self.current_player_turn
        
        for key in self.observation_space:
            if(isinstance(self.observation_space[key], spaces.MultiDiscrete)):
                if(not self.observation_space[key].contains(self.observation[key])):
                    too_high_indices = [i for i,x in enumerate(self.observation[key]) if x >= int(self.observation_space[key].nvec[i])]
                    for i in too_high_indices:
                        self.observation[key][i] = int(self.observation_space[key].nvec[i]) - 1
        
        return self.observation, reward, terminated, truncated, info
   
    def recordFrame(self):
        #TODO: make this function save and image for each frame instead of overwriting it. maybe even open a new folder for each run?
        save_path = 'last_frame.png'
        plt.savefig(save_path)
    
    def render(self, record_flag:bool = False):
        # the commented lines might be needed when we scale to more types of units. for now i print the infantries directly
        # nx.set_node_attributes(G, dict(zip(territories,obs['player1_infantry'])), name = 'p1_infantry')
        # nx.set_node_attributes(G, dict(zip(territories,obs['player2_infantry'])), name = 'p2_infantry')
        
        fig = plt.figure()
        G_p1_pos = self.G_node_pos.copy()
        G_p2_pos = self.G_node_pos.copy()
        for i, key in (self.G_node_pos.items()):
            G_p1_pos[i] = self.G_node_pos[i] + [0, 0.15]
            G_p2_pos[i] = self.G_node_pos[i] - [0, 0.15]

        nx.draw(self.G, self.G_node_pos, with_labels = True, 
                node_color = self.observation['territory_owner'], cmap = plt.get_cmap('jet'), node_size=500,
                font_color = "white", font_size = 15)

        nx.draw_networkx_labels(self.G,G_p1_pos, labels = dict(zip(territories,self.observation['player1_infantry'])), font_color = "blue")
        nx.draw_networkx_labels(self.G,G_p2_pos, labels = dict(zip(territories,self.observation['player2_infantry'])), font_color = "red")
        fig.canvas.draw()

        if(record_flag):
            self.recordFrame()

        if self.render_mode == "human":
            plt.show()
        elif self.render_mode == "rgb_array":
            return np.array(fig.canvas.renderer.buffer_rgba())

if __name__ == "__main__":
    game = AxisAndAlliesEnv(render_mode="human")

    obs, info = game.reset()
    print('first observation after reset: ', obs)
    game.render()
    num_of_steps = 0
    for i in range(num_of_steps):
        action = game.action_space.sample()
        print(f'random action {i+1}: ', action)
        observation, reward, terminated, truncated, info = game.step(action)

        game.render()

        if terminated or truncated:
            print(f'after step {i+1}: ', observation, reward, terminated)
            print('Game over..')
            break
        else:
            print(f'after step {i+1}: ', observation, reward, terminated)

