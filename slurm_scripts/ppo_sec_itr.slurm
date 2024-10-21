#!/bin/bash

#SBATCH --job-name=RL_second
#SBATCH --output=/home-mscluster/zschwark/RL_assignment/output_logs/PPO_second_iter_output_logs.out
#SBATCH -e /home-mscluster/zschwark/RL_assignment/output_logs/PPO_second_iter_err.err
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p stampede

source ~/.bashrc
conda activate rl_assignment
python3 ~/RL_assignment/ppo/recurrent_ppo.py