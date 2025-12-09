FLAGS = -I ~/miniconda3/envs/cpp_libs/include/ -arch=sm_80 --extended-lambda -rdc=true -std=c++17 -O2

build:
	nvcc $(FLAGS) main.cu utils.cu -o main

build_coada:
	sbatch -p dgxa100 -t 12:00:00 --output=output_build_coada.log --mem-per-cpu 32G --wrap="nvcc $(FLAGS) main.cu utils.cu -o main" && watch -n 1 squeue -u vlad_adrian.ulmeanu

build_test:
	nvcc $(FLAGS) $(FNAME).cu -o $(FNAME)

# FNAME = test_thrust / test_cub_segmented_scan 
