import gymnasium as gym
import grid2op
import sys
from stable_baselines3.common.monitor import Monitor
import tqdm
display_tqdm = False  # this is set to False for ease with the unitt test, feel free to set it to True
import wandb
sys.path.insert(0,"./")
from provided_wrapper.env import Gym2OpEnv
from sb3_contrib import RecurrentPPO
from wandb.integration.sb3 import WandbCallback
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    env = Gym2OpEnv(baseline=False, first_iteraion=False,second_iteraion=True)
    env = Monitor(env)
    
    print("#####################")
    print("#   ACTION SPACE    #")
    print("#####################")
    print(env.action_space)
    print("#####################\n\n")


    run = wandb.init(
        project="RL_project",
        name = "Recurrent_PPO",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
    )


    model = RecurrentPPO("MultiInputLstmPolicy",
                env,
                verbose=1,
                device=device,
                tensorboard_log=f"runs/{run.id}")
    
    
    model.learn(total_timesteps=25000, callback=WandbCallback(gradient_save_freq=100,model_save_path=f"models/{run.id}",verbose=2), progress_bar=True)

    run.finish()

    curr_step = 0
    curr_return = 0

    is_done = False
    obs, info = env.reset()
    print(f"step = {curr_step} (reset):")
    print(f"\t obs = {obs}")
    print(f"\t info = {info}\n\n")

    while not is_done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(action)

        curr_step += 1
        curr_return += reward
        is_done = terminated or truncated

        #print(f"step = {curr_step}: ")
        #print(f"\t obs = {obs}")
        #print(f"\t reward = {reward}")
        #print(f"\t terminated = {terminated}")
        #print(f"\t truncated = {truncated}")
        #print(f"\t info = {info}")

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
