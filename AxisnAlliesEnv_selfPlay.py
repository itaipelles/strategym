from AxisnAlliesEnv import AxisAndAlliesEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import copy
import numpy as np
import math

class AxisAndAlliesEnv_selfPlay(AxisAndAlliesEnv):
    action_space = AxisAndAlliesEnv.action_space
    observation_space = AxisAndAlliesEnv.observation_space
    AI_policy : PPO = None
    currentPlayer : int = 0

    def __init__(self):
        super().__init__()
        self.AI_policy = None
        self.currentPlayer = 0
        pass
    def step(self, action, render_mid_round:bool= False):
        observation_mid, reward_mid, terminated_mid, truncated_mid, info_mid = self.step_wrapper(action)
        if(render_mid_round):
            super().render()
        if(terminated_mid or truncated_mid or self.current_player_turn==self.currentPlayer):
            return self.adjust_observation(observation_mid), self.adjust_reward(reward_mid), terminated_mid, truncated_mid, info_mid
        
        observation_end, reward_end, terminated_end, truncated_end, info_end = self.step_wrapper(action)
        return self.adjust_observation(observation_end), self.adjust_reward(reward_mid + reward_end), (terminated_mid or terminated_end), (truncated_mid or truncated_end), info_end
    
    def step_wrapper(self, action):
        if self.current_player_turn == self.currentPlayer:
            observation, reward, terminated, truncated, info = super().step(action, add_penalties=True)
        else:
            action_AI = None
            if self.AI_policy is None:
                action_AI = self.action_space.sample()
            else:
                action_AI, _ = self.AI_policy.predict(self.adjust_observation(copy.deepcopy(self.observation),target_player=0))
            observation, reward, terminated, truncated, info = super().step(action_AI)
        return observation, reward, terminated, truncated, info
        
    def reset(self, seed=None, options=None):
        return super().reset()
    def render(self, record_flag:bool = False):
        super().render(record_flag)
    def adjust_observation(self, observation, target_player = 1):
        if(self.currentPlayer == target_player):
            temp = observation['player1_infantry']
            observation['player1_infantry'] = observation['player2_infantry']
            observation['player2_infantry'] = temp
            observation['territory_owner'] = 1 - observation['territory_owner']
        return observation
    def adjust_reward(self, reward):
        return np.sign(0.5-self.currentPlayer)*reward
    def swap_players(self):
        self.reset()
        self.currentPlayer = 1 - self.currentPlayer 
        self.AI_policy = PPO.load("model_snapshot")

class SelfPlayCallback(BaseCallback):
    env : VecEnv
    current_model : PPO
    def __init__(self, env, model, swap_interval = math.inf, save_interval = math.inf, *args, **kwargs):
        super(SelfPlayCallback, self).__init__(*args, **kwargs)
        self.env = env
        self.current_model = model
        if swap_interval < save_interval:
            save_interval = swap_interval
        self.swap_interval = swap_interval
        self.save_interval = save_interval
        self.rollout_counter = 0
    def _on_step(self) -> bool:
        return super()._on_step()    
    def _on_training_start(self) -> None:
        self.rollout_counter = 0
    def _on_rollout_start(self) -> None:
        self.rollout_counter += 1
        if self.rollout_counter%self.save_interval == 0:
            print('saving model snapshot')
            self.current_model.save("model_snapshot")
        if self.rollout_counter%self.swap_interval == 0:
            self.env.env_method('swap_players')
            

    
            