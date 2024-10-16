import gymnasium as gym
import grid2op
from stable_baselines3.common.monitor import Monitor
import wandb
import sys
sys.path.insert(0,"./") # this makes it possible to import the wrapper since the env.py is in it's own folder
from provided_wrapper.env import Gym2OpEnv
from stable_baselines3 import DQN
from wandb.integration.sb3 import WandbCallback
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sb3_contrib import QRDQN


def main():
    env = Gym2OpEnv()
    env = Monitor(env)
    
    run = wandb.init(
        project="RL_project",
        name = "QR-DQN_Baseline",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )
    
    
    model = DQN("MultiInputPolicy",
                env,
                verbose =1,
                device = device,
                tensorboard_log=f"runs/{run.id}",
                learning_rate = 1e-4,
                buffer_size = 25000,
                learning_starts = 1000,
                batch_size = 64,
                tau = 0.001,
                gamma = 0.99,
                train_freq = 1,
                gradient_steps = 1,
                target_update_interval = 1000,
                exploration_fraction = 0.1,
                exploration_initial_eps = 1.0,
                exploration_final_eps = 0.05,
                max_grad_norm =10)
    
    policy_kwargs = dict(n_quantiles=50)
    model2 = QRDQN("MultiInputPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
    model2.learn(total_timesteps=25000, log_interval=10, callback=WandbCallback(gradient_save_freq=100,model_save_path=f"models/{run.id}",verbose=2), progress_bar=True)
    
    
    
    #Training the model
    #model.learn(total_timesteps = 25000, log_interval = 10, callback=WandbCallback(gradient_save_freq=100,model_save_path=f"models/{run.id}",verbose=2), progress_bar=True)
    
    run.finish()
    
    #saving the model
    ##model.save("first_implementation_model")
    
    #Evaluating
    #mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    
   # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    #Run for a few episodes with trained agent
    obs = env.reset()
    
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic = True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            
    env.close()
    

if __name__ == "__main__":
    main()