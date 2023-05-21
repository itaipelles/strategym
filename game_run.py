from stable_baselines3 import PPO
from axisAndAllies_game.board import Players
from gym_env.game_env_selfPlay import AxisAndAlliesEnv_selfPlay

env = AxisAndAlliesEnv_selfPlay(render_mode="human")
env.set_player(Players.RUSSIA)
model_russia = PPO.load("models/RUSSIA_model_1", env=env,device='cpu')
model_germany = PPO.load("models/GERMANY_model_2", env=env,device='cpu')
policies = {Players.RUSSIA:model_russia, Players.GERMANY:model_germany}
env.set_policies(policies)
obs, _ = env.reset()

env.render()
for i in range(2000):
    action, _states = policies[env.currently_training_player].predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
        env.reset()
        env.render()