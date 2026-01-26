FLAGS = -I ~/miniconda3/envs/cpp_libs/include/ -arch=sm_80 --extended-lambda -rdc=true -std=c++17 -O2
LOCAL_FLAGS = -arch=sm_61 --extended-lambda -rdc=true -std=c++17 -O2

LOCAL_ARROW_I = -isystem /home/vlad/miniconda3/envs/cpp_libs/include
LOCAL_ARROW_L = -L /home/vlad/miniconda3/envs/cpp_libs/lib
LOCAL_ARROW_L_DYN = -larrow -lparquet -Xlinker -R,/home/vlad/miniconda3/envs/cpp_libs/lib

COADA_BASE = sbatch -p dgxa100 -t 12:00:00 --mem-per-cpu 32G
WATCH = watch -n 1 squeue -u vlad_adrian.ulmeanu


build:
	nvcc $(FLAGS) main.cu utils.cu -o main

build_coada:
	$(COADA_BASE) --output=output_build_coada.log --wrap="nvcc $(FLAGS) main.cu utils.cu -o main" && $(WATCH)

build_local:
	nvcc $(LOCAL_ARROW_I) $(LOCAL_ARROW_L) $(LOCAL_FLAGS) main.cu e3s_sawnoff.cu readers.cu utils.cu -o main $(LOCAL_ARROW_L_DYN)

build_local_debug:
	nvcc $(LOCAL_ARROW_I) $(LOCAL_ARROW_L) $(LOCAL_FLAGS) main.cu readers.cu utils.cu -o main $(LOCAL_ARROW_L_DYN)

build_test:
	nvcc $(FLAGS) $(FNAME).cu -o $(FNAME)

run_coada:
	$(COADA_BASE) --output=output_run_coada.log --gres gpu:1 --wrap="time ./main $(INFILE) $(OUTFILE)" && $(WATCH)

run_local_cses:
	time ./main $(OUTFILE) $(INFILE) && diff -s $(OUTFILE) $(EXPFILE)

# ! conda activate cpp_libs.

# FNAME = test_thrust / test_cub_segmented_scan
# INFILE = test_inputs/input0.txt
# EXPFILE = test_inputs/expected0.txt
# OUTFILE = test_inputs/output.txt

