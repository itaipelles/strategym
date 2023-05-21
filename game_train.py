from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from gym_env.game_env_selfPlay import AxisAndAlliesEnv_selfPlay, Players

if __name__ == "__main__":
    env = make_vec_env(AxisAndAlliesEnv_selfPlay, n_envs=8)
    
    policy_kwargs = dict(net_arch=[64, 64])
    model_russia = PPO('MultiInputPolicy', env, verbose=4, n_steps=512, batch_size=512, policy_kwargs=policy_kwargs, device='cpu')
    model_germany = PPO('MultiInputPolicy', env, verbose=4, n_steps=512, batch_size=512, policy_kwargs=policy_kwargs, device='cpu')
    policies = {Players.RUSSIA:model_russia, Players.GERMANY:model_germany}
    env.env_method('set_policies', **{'AI_policies' : policies})
    
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=90, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, eval_freq=10000, verbose=1)

    num_of_generations = 4
    for generation_num in range(num_of_generations):
        for player in Players:
            print(f'currently training: {player.name} generation: {generation_num}')
            checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./model_checkpoints/', name_prefix=f'{player.name}_model_{generation_num}')
            env.env_method('set_player', **{'player':player})
            env.reset()
            policies[player].learn(total_timesteps=500000, callback=[checkpoint_callback, eval_callback])
            policies[player].save(f'./models/{player.name}_model_{generation_num}')