from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from game_selfPlay import AxisAndAlliesEnv_selfPlay
from game_selfPlay import SelfPlayCallback
from game import AxisAndAlliesEnv
import time
import math

if __name__ == "__main__":
    env = make_vec_env(AxisAndAlliesEnv_selfPlay, n_envs=4)
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO("MultiInputPolicy", env, verbose=4, n_steps=512, batch_size=64, policy_kwargs=policy_kwargs, device='auto')
    # model = PPO.load("test_model", env=env)
    update_AI_callback = SelfPlayCallback(env,model, updateRate=math.inf)

    model.learn(total_timesteps=100000, callback=update_AI_callback)
    model.save("test_model")
    del model

    # run test
    env = AxisAndAlliesEnv_selfPlay()
    model = PPO.load("test_model", env=env)
    obs, _ = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, truncated, info = env.step(action, render_mid_round=True)
        env.render()