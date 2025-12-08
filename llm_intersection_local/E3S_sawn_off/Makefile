FLAGS = -I ~/miniconda3/envs/cpp_libs/include/ --extended-lambda -std=c++17 -O2

build:
	nvcc $(FLAGS) main.cu utils.cu -o main

build_test:
	nvcc $(FLAGS) $(FNAME).cu -o $(FNAME)

# FNAME = test_thrust / test_cub_segmented_scan 
