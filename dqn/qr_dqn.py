from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict
import numpy as np
from dqn_env import Gym2OpEnv
import gymnasium as gym
import grid2op
from stable_baselines3.common.monitor import Monitor
import wandb
import sys
sys.path.insert(0,"./") # this makes it possible to import the wrapper since the env.py is in it's own folder
#from provided_wrapper.env import Gym2OpEnv
from stable_baselines3 import DQN
from wandb.integration.sb3 import WandbCallback
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sb3_contrib import QRDQN
import numpy as np

keys_array = [
    'a_ex', 'a_or', 'active_alert', 'actual_dispatch', 'alert_duration', 'attack_under_alert',
    'attention_budget', 'current_step', 'curtailment', 'curtailment_limit',
    'curtailment_limit_effective', 'curtailment_limit_mw', 'curtailment_mw', 'day',
    'day_of_week', 'delta_time', 'duration_next_maintenance', 'gen_margin_down',
    'gen_margin_up', 'gen_p', 'gen_p_before_curtail', 'gen_q', 'gen_theta', 'gen_v',
    'hour_of_day', 'is_alarm_illegal', 'last_alarm', 'line_status', 'load_p', 'load_q',
    'load_theta', 'load_v', 'max_step', 'minute_of_hour', 'month', 'p_ex', 'p_or',
    'prod_p', 'prod_q', 'prod_v', 'q_ex', 'q_or', 'rho', 'storage_charge',
    'storage_power', 'storage_power_target', 'storage_theta', 'target_dispatch',
    'thermal_limit', 'theta_ex', 'theta_or', 'time_before_cooldown_line',
    'time_before_cooldown_sub', 'time_next_maintenance', 'time_since_last_alarm',
    'time_since_last_alert', 'time_since_last_attack', 'timestep_overflow', 'topo_vect',
    'total_number_of_alert', 'v_ex', 'v_or', 'was_alarm_used_after_game_over',
    'was_alert_used_after_attack', 'year'
]
attr_to_keep_3 = ['a_ex', 'a_or', 'active_alert', 'actual_dispatch', 'alert_duration',
    'attention_budget', 'current_step','curtailment', 'curtailment_limit', 'curtailment_limit_effective', 'curtailment_limit_mw', 'curtailment_mw','day',
    'day_of_week', 'delta_time', 'duration_next_maintenance', 'gen_margin_down',
    'gen_margin_up', 'gen_p', 'gen_p_before_curtail', 'gen_q', 'gen_theta', 'gen_v',
    'hour_of_day', 'last_alarm', 'line_status', 'load_p', 'load_q',
    'load_theta', 'load_v', 'max_step', 'minute_of_hour', 'month', 'p_ex', 'p_or',
    'prod_p', 'prod_q', 'prod_v', 'q_ex', 'q_or', 'rho', 'storage_charge',
    'storage_power', 'storage_power_target', 'storage_theta', 'target_dispatch',
    'thermal_limit', 'theta_ex', 'theta_or', 'time_before_cooldown_line',
    'time_before_cooldown_sub', 'time_next_maintenance',
    'timestep_overflow', 'topo_vect','v_ex', 'v_or',
     'year']#so far the best
attr_to_keep_4 = ['a_ex', 'a_or',  'actual_dispatch',
    'attention_budget', 'current_step','curtailment', 'curtailment_limit', 'curtailment_limit_effective', 'curtailment_limit_mw', 'curtailment_mw','day',
    'day_of_week', 'delta_time', 'duration_next_maintenance', 'gen_margin_down',
    'gen_margin_up', 'gen_p', 'gen_p_before_curtail', 'gen_q', 'gen_theta', 'gen_v',
    'hour_of_day', 'line_status', 'load_p', 'load_q',
    'load_theta', 'load_v', 'max_step', 'minute_of_hour', 'month', 'p_ex', 'p_or',
    'prod_p', 'prod_q', 'prod_v', 'q_ex', 'q_or', 'rho', 'storage_charge',
    'storage_power', 'storage_power_target', 'storage_theta', 'target_dispatch',
    'thermal_limit', 'theta_ex', 'theta_or', 'time_before_cooldown_line',
    'time_before_cooldown_sub', 'time_next_maintenance',
    'timestep_overflow', 'topo_vect','v_ex', 'v_or',
     'year'] #this one is bad so active alert and alert duration is important
attr_to_keep_5 = ['a_ex', 'a_or', 'active_alert', 'actual_dispatch', 'alert_duration',
    'attention_budget', 'current_step','curtailment', 'curtailment_limit', 'curtailment_limit_effective', 'curtailment_limit_mw', 'curtailment_mw','day',
    'day_of_week', 'delta_time', 'duration_next_maintenance', 'gen_margin_down',
    'gen_margin_up', 'gen_p', 'gen_p_before_curtail', 'gen_q', 'gen_theta', 'gen_v',
    'hour_of_day', 'last_alarm', 'line_status', 'load_p', 'load_q',
    'load_theta', 'load_v', 'max_step', 'minute_of_hour', 'month', 'p_ex', 'p_or',
    'prod_p', 'prod_q', 'prod_v', 'q_ex', 'q_or', 'rho',
    'storage_power', 'storage_power_target', 'storage_theta', 'target_dispatch',
    'thermal_limit', 'theta_ex', 'theta_or', 'time_before_cooldown_line',
    'time_before_cooldown_sub', 'time_next_maintenance',
    'timestep_overflow', 'topo_vect','v_ex', 'v_or',
     'year'] #removed storage charged - does very bad
attr_to_keep_6 = ['a_ex', 'a_or', 'active_alert', 'actual_dispatch', 'alert_duration',
    'attention_budget', 'current_step','curtailment', 'curtailment_limit', 'curtailment_limit_effective', 'curtailment_limit_mw', 'curtailment_mw','day',
    'day_of_week', 'delta_time', 'duration_next_maintenance', 'gen_margin_down',
    'gen_margin_up', 'gen_p', 'gen_p_before_curtail', 'gen_q', 'gen_theta', 'gen_v',
    'hour_of_day', 'last_alarm', 'line_status', 'load_p', 'load_q',
    'load_theta', 'load_v', 'max_step', 'minute_of_hour', 'month', 'p_ex', 'p_or',
    'prod_p', 'prod_q', 'prod_v', 'q_ex', 'q_or', 'rho', 'storage_charge',
    'storage_power', 'storage_power_target', 'storage_theta', 'target_dispatch',
    'thermal_limit', 'theta_ex', 'theta_or', 'time_before_cooldown_line',
     'time_next_maintenance',
    'timestep_overflow', 'topo_vect','v_ex', 'v_or',
     'year']#'time_before_cooldown_sub'
attr_to_keep = ["rho", "gen_p", "load_p", "topo_vect", "actual_dispatch"]


# Custom Action Tracker Callback
class ActionTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ActionTrackerCallback, self).__init__(verbose)
        self.action_counter = defaultdict(int)

    def _on_step(self):
        # Track the actions taken during training
        action = self.locals['actions']
        if isinstance(action, np.ndarray):
            action_key = tuple(action)
        else:
            action_key = (action,)
        self.action_counter[action_key] += 1
        return True

    def _on_training_end(self):
        # Called at the end of training - print action distribution here
        print("Training completed! Action distribution:")
        for action, freq in sorted(self.action_counter.items(), key=lambda x: x[1], reverse=True):
            print(f"Action: {action}, Frequency: {freq}")

    def get_action_distribution(self):
        return dict(self.action_counter)
    
def evaluate_model(model, env, num_episodes=10):
    all_episode_rewards = []
    all_episode_lengths = []
    all_n1_rewards = []
    all_l2rpn_rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        n1_reward = 0
        l2rpn_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            # Extract individual rewards
            n1_reward += info.get('rewards', {}).get('N1', 0)
            l2rpn_reward += info.get('rewards', {}).get('L2RPN', 0)

        all_episode_rewards.append(episode_reward)
        all_episode_lengths.append(episode_length)
        all_n1_rewards.append(n1_reward)
        all_l2rpn_rewards.append(l2rpn_reward)

    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    mean_length = np.mean(all_episode_lengths)
    std_length = np.std(all_episode_lengths)
    mean_n1 = np.mean(all_n1_rewards)
    std_n1 = np.std(all_n1_rewards)
    mean_l2rpn = np.mean(all_l2rpn_rewards)
    std_l2rpn = np.std(all_l2rpn_rewards)

    return mean_reward, std_reward, mean_length, std_length, mean_n1, std_n1, mean_l2rpn, std_l2rpn


def main():
    # Initialize environment and monitor
    env = Gym2OpEnv(attr_to_keep_3)
    env = Monitor(env)

    # Initialize Wandb
    run = wandb.init(
        project="RL_project",
        name="DQN improvement 2 qr-dqn",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )

    # Initialize DQN model
    # model = DQN("MlpPolicy", env, verbose=1, device = device, tensorboard_log="./dqn_grid2op_tensorboard/")
    policy_kwargs = dict(n_quantiles=50)
    model2 = QRDQN("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device=device)


    # Initialize custom callback to track actions
    action_callback = ActionTrackerCallback()

    # Training the model with Wandb and custom action tracking callback
    model2.learn(
        total_timesteps=100000,
        log_interval=10,
        callback=[WandbCallback(gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2), action_callback],
        progress_bar=True
    )

    # After training, print the action distribution
    action_distribution = action_callback.get_action_distribution()

    print("\nFinal Action Distribution:")
    for action, freq in sorted(action_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"Action: {action}, Frequency: {freq}")


    #Evaluating the trained model
    print("\nEvaluating the trained model")
    mean_reward, std_reward, mean_length, std_length, mean_n1, std_n1, mean_l2rpn, std_l2rpn = evaluate_model(model2, env, num_episodes=50)
    
    #wandb.log({
    #    "eval/mean_total_reward": mean_reward,
    #    "eval/std_total_reward": std_reward,
    #    "eval/mean_episode_length": mean_length,
    #    "eval/std_episode_length": std_length,
    #    "eval/mean_N1_reward": mean_n1,
    #    "eval/std_N1_reward": std_n1,
    #    "eval/mean_L2RPN_reward": mean_l2rpn,
    #    "eval/std_L2RPN_reward": std_l2rpn
    #})

    print(f"\nEvaluation Results:")
    print(f"Mean Total Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f} +/- {std_length:.2f}")
    print(f"Mean N1 Reward: {mean_n1:.2f} +/- {std_n1:.2f}")
    print(f"Mean L2RPN Reward: {mean_l2rpn:.2f} +/- {std_l2rpn:.2f}")
    run.finish()

    # # Test the trained model
    # obs, _ = env.reset()  # Get initial observation and info
    # for _ in range(1000):
    #     # Predict action using only the observation part of the returned tuple
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, _, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs, _ = env.reset()  # Reset environment if done, and unpack the tuple again

    env.close()


if __name__ == "__main__":
    main()