FLAGS = -I ~/miniconda3/envs/cpp_libs/include/ -arch=sm_80 --extended-lambda -rdc=true -std=c++17 -O2
COADA_BASE = sbatch -p dgxa100 -t 12:00:00 --mem-per-cpu 32G

build:
	nvcc $(FLAGS) main.cu utils.cu -o main

build_coada:
	$(COADA_BASE) --output=output_build_coada.log --wrap="nvcc $(FLAGS) main.cu utils.cu -o main" && watch -n 1 squeue -u vlad_adrian.ulmeanu

build_test:
	nvcc $(FLAGS) $(FNAME).cu -o $(FNAME)

run_coada:
	$(COADA_BASE) --output=output_run_coada.log --gres gpu:1 --wrap="time ./main $(INFILE) $(OUTFILE)"

# FNAME = test_thrust / test_cub_segmented_scan
# INFILE = test_inputs/input.txt
# OUTFILE = test_inputs/output.txt
