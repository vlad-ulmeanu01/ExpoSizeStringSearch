#!/bin/bash
#SBATCH -p dgxa100
#SBATCH -t 12:00:00
#SBATCH --output=output.log
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

VENV_PATH="$HOME/uv_envs/hugg/bin"

source $VENV_PATH/activate
time $VENV_PATH/python pythia.py



