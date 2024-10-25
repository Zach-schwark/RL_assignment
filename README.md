# RL_assignment
Power Grid Assignment for the Reinforcement Learning course for Wits Computer Science Honours.

## Student Numbers:
- Ashlea Smith: 2455744
- Zach Schwark: 2434346

## File Structure:

### DQN:
- The DQN files are in the folder "dqn".
- The DQN files are all notebook files that can be run on Google Colab or locally.
- Each DQN iteration is its own notebook.
- These notebook files contain the "env.py" code in cells in the notebooks.

### PPO:
- The PPO files are in the folder "ppo"
- The PPO models use the "env.py" file that is stored in "provided_wrapper".
- The Baseline and Improvement 1 both use "ppo.py"
- Improvement 2 uses "recurrent_ppo.py"

#### Running PPO: 
- "ppo.py" is setup to run using job scripts, thereore this file can be executed in the terminal. Please ensure that you specify either "baseline" or "first" as arguments when running
this file in the terminal, as this decided which code to use for the different iterations.
- "recurrent_ppo.py" can also be run via jobscripts or the terminal, but it can be run without specifying argugments in the terminal.

### slurm_scripts:
- This folder contains the slurm scripts for running the PPO code on a cluster.

### output_logs:
- This folder is where the output logs are written to when running the slurm jobs.

### Python Environment:
- The python libaries and versions needed are in "requirements.txt"
- Please note: Python version 3.12.0 was used.

## Useful Links:

- [Github for Grid2OP](https://github.com/rte-france/Grid2Op)
- [Documentation for Grid2Op](https://grid2op.readthedocs.io/en/latest/)
- [Grid2Op observation space](https://grid2op.readthedocs.io/en/latest/observation.html)
- [Grid2Op action space](https://grid2op.readthedocs.io/en/latest/action.html)
- [Notebook 11: Grid2Op Getting Started](https://github.com/rte-france/Grid2Op/blob/c71a2dfb824dae7115394266e02cc673c8633a0e/getting_started/11_IntegrationWithExistingRLFrameworks.ipynb)
- [Stable Baselines 3 Github](https://github.com/DLR-RM/stable-baselines3)
- [Stable Baselines 3 Docs](https://stable-baselines3.readthedocs.io/en/master/)