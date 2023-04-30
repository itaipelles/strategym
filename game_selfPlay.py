from game import AxisAndAlliesEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import copy

class AxisAndAlliesEnv_selfPlay(AxisAndAlliesEnv):
    action_space = AxisAndAlliesEnv.action_space
    observation_space = AxisAndAlliesEnv.observation_space
    AI_policy:PPO

    def __init__(self):
        super().__init__()
        self.AI_policy = None
        pass
    def step(self, action, render_mid_round:bool= False):
        observation, reward, terminated, truncated, info = super().step(action)
        if(render_mid_round):
            super().render()
            
        action_AI = None
        if self.AI_policy is None:
            action_AI = self.action_space.sample()
        else:
            action_AI, _ = self.AI_policy.predict(observation)

        new_observation, new_reward, new_terminated, new_truncated, new_info = super().step(action_AI)
        return new_observation, (reward + new_reward), (terminated or new_terminated), (truncated or new_truncated), info
        
    def reset(self):
        return super().reset()
    def render(self, record_flag:bool = False):
        super().render(record_flag)

class SelfPlayCallback(BaseCallback):
    env : AxisAndAlliesEnv_selfPlay
    model : PPO
    def __init__(self, env, model, updateRate, *args, **kwargs):
        super(SelfPlayCallback, self).__init__(*args, **kwargs)
        self.model = model
        self.env = env
        self.updateRate = updateRate
    def _on_step(self) -> bool:
        result = super(SelfPlayCallback, self)._on_step()
        if self.num_timesteps%self.updateRate == 0:
            # this line crushes, needs an alternative
            self.env.AI_policy = copy.deepcopy(self.model)
        return result