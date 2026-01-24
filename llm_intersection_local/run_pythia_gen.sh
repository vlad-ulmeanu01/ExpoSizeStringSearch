#!/bin/bash
#SBATCH -p dgxh100
#SBATCH -t 12:00:00
#SBATCH --output=output2.log
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

VENV_PATH="$HOME/uv_envs/hugg_local/bin"

source $VENV_PATH/activate
time $VENV_PATH/python pythia.py
# time python pythia.py



