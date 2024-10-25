#!/bin/bash

#SBATCH --job-name=RL_first
#SBATCH --output=/home-mscluster/zschwark/RL_assignment/output_logs/PPO_first_iter_output_logs.txt
#SBATCH -N 1
#SBATCH -p stampede

source ~/.bashrc
conda activate rl_assignment
python3 ~/RL_assignment/ppo/ppo.py first
