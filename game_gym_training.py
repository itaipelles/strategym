import gym
import json
import datetime as dt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from game import AxisAndAlliesEnv
import pandas as pd
import time

if __name__ == "__main__":
    # our env obs and actions are not properly compatible yet, run this script to see problems
    # also, DummyVecEnv vs make_vec_env??
    env = make_vec_env(AxisAndAlliesEnv, n_envs=4) 

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)

    obs = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        time.sleep(0.5)
