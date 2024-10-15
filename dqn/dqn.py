import gymnasium as gym
import grid2op
import sys
sys.path.insert(0,"./") # this makes it possible to import the wrapper since the env.py is in it's own folder
from provided_wrapper.env import Gym2OpEnv
import stable_baselines3

def main():
    env = Gym2OpEnv()
    
    check_env(env)
    
    model = DQN("MlpPolicy",
                env,
                verbose =1,
                learning_rate = 1e-4,
                buffer_size = 100000,
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
                max_grad_norm =10,
                douple_q = False)
    
    #Training the model
    model.learn(total_timesteps = 100000, log_interval = 10)
    
    #saving the model
    model.save("first_implementation_model")
    
    #Evaluating
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    #Run for a few episodes with trained agent
    obs = env.reset()
    
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic = True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            
    env.close()