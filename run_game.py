from stable_baselines3 import PPO
from gym_training.axisAndAllies_game_env_selfPlay import AxisAndAlliesEnv_selfPlay, Players

env = AxisAndAlliesEnv_selfPlay(Players.RUSSIA)
model_russia = PPO.load("models/RUSSIA_model_1", env=env,device='cpu')
model_germany = PPO.load("models/GERMANY_model_2", env=env,device='cpu')
policies = {Players.RUSSIA:model_russia, Players.GERMANY:model_germany}
env.set_policies(policies)
obs, _ = env.reset()

env.render()
for i in range(2000):
    action, _states = policies[env.currently_training_player].predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action, render_mid_round=True)
    env.render()
    if terminated:
        env.reset()
        env.render()