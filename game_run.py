from stable_baselines3 import PPO
from AxisnAlliesEnv_selfPlay import AxisAndAlliesEnv_selfPlay

env = AxisAndAlliesEnv_selfPlay()
env.currentPlayer = 0
env.AI_policy = PPO.load("game_model", env=env,device='cpu')
model = PPO.load("game_model", env=env,device='cpu')
obs, _ = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action, render_mid_round=True)
    env.render()