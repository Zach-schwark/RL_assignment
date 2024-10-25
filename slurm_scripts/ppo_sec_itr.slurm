#!/bin/bash

#SBATCH --job-name=RL_second
#SBATCH --output=/home-mscluster/zschwark/RL_assignment/output_logs/PPO_second_iter_output_logs.txt
#SBATCH -N 1
#SBATCH -p stampede

source ~/.bashrc
conda activate rl_assignment
python3 ~/RL_assignment/ppo/recurrent_ppo.py
