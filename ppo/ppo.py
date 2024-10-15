import gymnasium as gym
import grid2op
import sys
from stable_baselines3.common.monitor import Monitor
import wandb
sys.path.insert(0,"./")
from provided_wrapper.env import Gym2OpEnv
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# this is a very basic of the initial implementaion of PPO.
# I used the random agent code in "env.py" as a template to help use PPO to "step" through the environment.

# Note: i dont think this initial implementation is finished yet. I have just gotten the PPO "learn" function to run with our errors and to use that agent to step through the environment without erros.
# I followed the simple stable baselines example of PPO to implement it, therefore there is potentiallyy more that can be and needs to bedoen regarding PPO, this was just to get it to runn with out errors etc.



# evaluvation plots i want to do:

# rewards over episodes - averaged over 100 runs
# steps per episode
# losses per episodes


def main():

    max_steps = 3

    env = Gym2OpEnv()
    env = Monitor(env)
    
    print("#####################")
    print("# OBSERVATION SPACE #")
    print("#####################")
    print(env.observation_space)
    print("#####################\n")

    print("#####################")
    print("#   ACTION SPACE    #")
    print("#####################")
    print(env.action_space)
    print("#####################\n\n")
    

    run = wandb.init(
        project="RL_project",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )



    model = PPO("MultiInputPolicy", env, verbose=1, device=device, tensorboard_log=f"runs/{run.id}")
    model.learn(total_timesteps=25000, callback=WandbCallback(gradient_save_freq=100,model_save_path=f"models/{run.id}",verbose=2), progress_bar=True)

    run.finish()

    curr_step = 0
    curr_return = 0

    is_done = False
    obs, info = env.reset()
    print(f"step = {curr_step} (reset):")
    print(f"\t obs = {obs}")
    print(f"\t info = {info}\n\n")

        
    while not is_done and curr_step < max_steps:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        curr_step += 1
        curr_return += reward
        is_done = terminated or truncated

        print(f"step = {curr_step}: ")
        print(f"\t obs = {obs}")
        print(f"\t reward = {reward}")
        print(f"\t terminated = {terminated}")
        print(f"\t truncated = {truncated}")
        print(f"\t info = {info}")

        is_action_valid = not (info["is_illegal"] or info["is_ambiguous"])
        print(f"\t is action valid = {is_action_valid}")
        if not is_action_valid:
            print(f"\t\t reason = {info['exception']}")
        print("\n")

    print("###########")
    print("# SUMMARY #")
    print("###########")
    print(f"return = {curr_return}")
    print(f"total steps = {curr_step}")
    print("###########") 
       

if __name__ == "__main__":
    main()
