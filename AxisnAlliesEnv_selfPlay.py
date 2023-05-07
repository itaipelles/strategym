from AxisnAlliesEnv import AxisAndAlliesEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import copy

class AxisAndAlliesEnv_selfPlay(AxisAndAlliesEnv):
    action_space = AxisAndAlliesEnv.action_space
    observation_space = AxisAndAlliesEnv.observation_space
    AI_policy : PPO = None

    def __init__(self):
        super().__init__()
        self.AI_policy = None
        pass
    def step(self, action, render_mid_round:bool= False):
        observation_mid, reward_mid, terminated_mid, truncated_mid, info_mid = self.step_wrapper(action)
        if(render_mid_round):
            super().render()
        
        if(terminated_mid or truncated_mid):
            return observation_mid, reward_mid, terminated_mid, truncated_mid, info_mid

        observation_end, reward_end, terminated_end, truncated_end, info_end = self.step_wrapper(action)

        return observation_end, (reward_mid + reward_end), (terminated_mid or terminated_end), (truncated_mid or truncated_end), info_end
    
    def step_wrapper(self, action):
        if self.current_player_turn == 0:
            observation, reward, terminated, truncated, info = super().step(action)
        elif self.current_player_turn == 1:
            action_AI = None
            if True or (self.AI_policy is None):
                action_AI = self.action_space.sample()
            else:
                action_AI, _ = self.AI_policy.predict(observation)
            observation, reward, terminated, truncated, info = super().step(action_AI)
        return observation, reward, terminated, truncated, info
        
    def reset(self, seed=None, options=None):
        return super().reset()
    def render(self, record_flag:bool = False):
        super().render(record_flag)

class SelfPlayCallback(BaseCallback):
    env : AxisAndAlliesEnv_selfPlay
    current_model : PPO
    def __init__(self, env, model, updateInterval : int, *args, **kwargs):
        super(SelfPlayCallback, self).__init__(*args, **kwargs)
        self.env = env
        self.env.AI_policy = copy.deepcopy(model)
        self.current_model = model
        self.updateInterval = updateInterval
        self.rollout_counter = 0
    def _on_step(self) -> bool:
        return super()._on_step()    
    def _on_training_start(self) -> None:
        self.rollout_counter = 0
    def _on_rollout_start(self) -> None:
        self.rollout_counter += 1
        if self.rollout_counter%self.updateInterval == 0:
            # self.swap_players()
            # self.env.reset()
            print('saving model snapshot')
            self.current_model.save("model_snapshot")
            # self.env.AI_policy = PPO.load("model_snapshot")

    def swap_players(self):
        self.env.starting_player = 1 - self.env.starting_player
        self.env.opening_observation['territory_owner'] = 1 - self.env.opening_observation['territory_owner']
            