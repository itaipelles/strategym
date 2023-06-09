from stable_baselines3 import PPO
from gym_env.game_env import AxisAndAlliesEnv, Players
import numpy as np

class AxisAndAlliesEnvSelfPlay(AxisAndAlliesEnv):
    AI_policies:dict[Players,PPO]
    currently_training_player:Players

    def __init__(self, AI_policies:dict = {}, render_mode = None):
        super().__init__()
        self.AI_policies = AI_policies
        for player in Players:
            if(player not in self.AI_policies):
                self.AI_policies[player] = None
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        observation, info = super().reset()
        if self.game.current_player_turn != self.currently_training_player:
            observation, _, _, _, info = self.run_AI_turns()
        return observation, info

    def generate_AI_action(self):
        if(self.AI_policies[self.game.current_player_turn] is not None):
            return self.AI_policies[self.game.current_player_turn].predict(self.get_observation())[0]
        return self.action_space.sample()
        
    def step(self, action):
        board_value_prev = self.game.board.board_scores()
        observation, player_reward, terminated, truncated, info = self.training_player_step(action)
        if terminated:
            return observation, player_reward, terminated, truncated, info
        player_reward += -1*(1 + info['Illegal_Moves'])
        observation, AI_reward, terminated, truncated, info = self.run_AI_turns()
        round_reward = self.calc_networth_gain(board_value_prev)
        return observation, player_reward + AI_reward + round_reward, terminated, truncated, info
    
    def training_player_step(self,action):
            return super().step(action)
    
    def run_AI_turns(self):
        cummulative_reward = 0
        while(self.game.current_player_turn != self.currently_training_player ):
            if(self.render_mode == "human"):
                self.render()
            action = self.generate_AI_action()
            reward_sign = 1 if self.game.are_allies(self.game.current_player_turn, self.currently_training_player) else -1
            observation, reward, terminated, truncated, info = super().step(action)
            cummulative_reward += reward_sign*reward
            if terminated or truncated:
                return observation, cummulative_reward, terminated, truncated, info        
        return observation, cummulative_reward, terminated, truncated, info

    def calc_networth_gain(self, board_value_prev):
        board_value_current = self.game.board.board_scores()
        board_value_diff = board_value_current - board_value_prev
        round_reward = np.sum([value if self.game.are_allies(self.currently_training_player, Players(i)) else -value for i,value in enumerate(board_value_diff)])/10
        return round_reward
    
    def render(self):
        super().render()

    def set_policies(self, AI_policies:dict = {}):
        self.AI_policies = AI_policies

    def set_player(self, player:Players):
        self.currently_training_player = player