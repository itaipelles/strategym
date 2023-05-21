import gymnasium as gym
from gymnasium import spaces
import math
# from .. import run_game
from axisAndAllies_game.board import Players
from axisAndAllies_game.gameRenderer import GameRenderer 
from axisAndAllies_game.game import Game, set_game_v2, set_game

class AxisAndAlliesEnv(gym.Env):
    observation_space:spaces.Dict
    action_space:spaces.Box
    game:Game
    def __init__(self):
        self.game = set_game()
        self.renderer = GameRenderer(self.game.board)
        self.observation_space = spaces.Dict({
        'player1_infantry': spaces.Box(low=0.0, high=math.inf, shape=(self.game.board.num_of_territories(),)),
        'player2_infantry': spaces.Box(low=0.0, high=math.inf, shape=(self.game.board.num_of_territories(),)),
        'territory_owner': spaces.MultiBinary(self.game.board.num_of_territories()),
        })
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.game.board.num_of_adjacencies(),))
        return
    def reset(self, seed=None, options=None):
        self.game.reset()
        info = {}
        observation = self.get_observation()
        return observation, info
    
    def render(self):
        self.renderer.render(self.game.board,self.game.current_move,self.game.illegal_moves_count>0)
        return
    
    def step(self, action):
        info = {}
        truncated = False
        terminated = False
        reward = 0

        victory, illegal_moves = self.game.play_turn(action)
        observation = self.get_observation()
        info['Illegal_Moves'] = illegal_moves

        if victory:
            terminated = True
            reward = 100
        
        if self.game.round_counter > 100:
            truncated = True

        return observation, reward, terminated, truncated, info
    
    def get_observation(self):
        observation = {
        'player1_infantry': self.game.board.get_player_infantry(Players.RUSSIA),
        'player2_infantry': self.game.board.get_player_infantry(Players.GERMANY),
        'territory_owner': self.game.board.get_owners()}
        return observation