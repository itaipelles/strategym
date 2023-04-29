from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from game_selfPlay import AxisAndAlliesEnv_selfPlay
from game_selfPlay import SelfPlayCallback
from game import AxisAndAlliesEnv
import time
import math

if __name__ == "__main__":
    env = make_vec_env(AxisAndAlliesEnv_selfPlay, n_envs=4) 
    model = PPO("MultiInputPolicy", env, verbose=1)
    update_AI_callback = SelfPlayCallback(env,model, updateRate=math.inf)

    model.learn(total_timesteps=200000, callback=update_AI_callback)
    model.save("test_model")
    del model

    # run test
    model = PPO.load("test_model", env=env)
    obs = env.reset()
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        time.sleep(0.5)
