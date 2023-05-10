import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from AxisnAlliesEnv_selfPlay import AxisAndAlliesEnv_selfPlay
from AxisnAlliesEnv_selfPlay import SelfPlayCallback
import time
import math

if __name__ == "__main__":
    env = make_vec_env(AxisAndAlliesEnv_selfPlay, n_envs=8)
    policy_kwargs = dict(net_arch=[64, 64])
    model = PPO('MultiInputPolicy', env, verbose=4, n_steps=512, batch_size=512, policy_kwargs=policy_kwargs, device='cpu')
    # model = PPO.load("model_snapshot", env=env)
    update_AI_callback = SelfPlayCallback(env,model, swap_interval=10)

    model.learn(total_timesteps=1000000, callback=update_AI_callback)
    model.save("game_model")
    del model