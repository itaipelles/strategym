import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import math
from axisAndAllies_game_env import AxisAndAlliesEnv, Players

class AxisAndAlliesEnv_selfPlay(AxisAndAlliesEnv):
    AI_policies:dict[Players,PPO]
    currently_training_player:Players

    def __init__(self, training_player:Players = None, AI_policies:dict = {}):
        super().__init__()
        self.currently_training_player = training_player
        self.AI_policies = AI_policies

    def reset(self, seed=None, options=None):
        observation, info = super().reset()
        while(self.game.current_player_turn != self.currently_training_player):
            action = self.action_wrapper()
            observation, reward, terminated, truncated, info = super().step(action)

        return observation, info

    def action_wrapper(self,action = None):
        if(self.game.current_player_turn == self.currently_training_player):
            return action
        if(self.game.current_player_turn in self.AI_policies):
            return self.AI_policies[self.game.current_player_turn].predict(self.get_observation())[0]
        return self.action_space.sample()
        
    def step(self, action, render_mid_round:bool = False):
        observation, reward, terminated, truncated, info = super().step(action)
        reward += -1*(1 + info['Illegal_Moves'])
        
        while(self.game.current_player_turn != self.currently_training_player and not (terminated or truncated)):
            if(render_mid_round):
                self.render()
            action = self.action_wrapper()
            reward_sign = 1
            if not self.game.are_allies(self.game.current_player_turn, self.currently_training_player):
                reward_sign = -1

            observation, temp_reward, terminated, truncated, info = super().step(action)
            reward += reward_sign*temp_reward
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        super().render()

    def set_policies(self, AI_policies:dict = {}):
        self.AI_policies = AI_policies

    def set_player(self, player:Players):
        self.currently_training_player = player