#!/bin/bash

#SBATCH --job-name=RL_base
#SBATCH --output=/home-mscluster/zschwark/RL_assignment/output_logs/PPO_baseline_output_logs.txt
#SBATCH -N 1
#SBATCH -p stampede

source ~/.bashrc
conda activate rl_assignment
python3 ~/RL_assignment/ppo/ppo.py baseline
