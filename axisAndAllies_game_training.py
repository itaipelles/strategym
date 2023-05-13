from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from axisAndAllies_game_env_selfPlay import AxisAndAlliesEnv_selfPlay, Players
import math

if __name__ == "__main__":
    env = make_vec_env(AxisAndAlliesEnv_selfPlay, n_envs=8, env_kwargs={'training_player':Players.RUSSIA})
    
    policy_kwargs = dict(net_arch=[64, 64])
    model_russia = PPO('MultiInputPolicy', env, verbose=4, n_steps=512, batch_size=512, policy_kwargs=policy_kwargs, device='cpu')
    model_germany = PPO('MultiInputPolicy', env, verbose=4, n_steps=512, batch_size=512, policy_kwargs=policy_kwargs, device='cpu')
    policies = {Players.RUSSIA:model_russia, Players.GERMANY:model_germany}
    env.env_method('set_policies', **{'AI_policies' : policies})
    
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=90, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, eval_freq=10000, verbose=1)

    for i in range(4):
        for player in Players:
            print(f'currently training: {player.name} generation: {i}')
            checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./model_checkpoints/', name_prefix=f'{player.name}_model_{i}')
            env.env_method('set_player', **{'player':player})
            env.reset()
            policies[player].learn(total_timesteps=500000, callback=[checkpoint_callback, eval_callback])
            policies[player].save(f'./models/{player.name}_model_{i}')