#!/bin/sh

# conda activate hugg &&
# sbatch -p dgxh100 -t 12:00:00 --mem-per-cpu 32G ./pywrapper.sh &&
# watch -n 1 squeue -u vlad_adrian.ulmeanu

# sbatch -p xl -t 12:00:00 --gres gpu:1 --mem-per-cpu 32G ./pywrapper.sh &&
# sbatch -p dgxa100 -t 12:00:00 --gres gpu:1 --mem-per-cpu 32G ./pywrapper.sh &&
# watch -n 1 squeue -u vlad_adrian.ulmeanu

sbatch -p dgxh100 -t 12:00:00 --output=output.log --gres gpu:1 --mem-per-cpu 32G --wrap="time python pythia_gen.py" &&
watch -n 1 squeue -u vlad_adrian.ulmeanu

# cat timp a luat rularea jobului:
# sacct -j <myid> --format=Elapsed
