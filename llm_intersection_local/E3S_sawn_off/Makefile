FLAGS = -arch=sm_80 --extended-lambda -rdc=true -std=c++17 -O2
LOCAL_FLAGS = -arch=sm_61 --extended-lambda -rdc=true -std=c++17 -O2

ENV_PATH = /export/home/acs/stud/v/vlad_adrian.ulmeanu/.conda/envs/cpp_libs2
LOCAL_ENV_PATH = /home/vlad/miniconda3/envs/cpp_libs

ARROW_I = -isystem $(ENV_PATH)/include
ARROW_L = -L $(ENV_PATH)/lib
ARROW_L_DYN = -larrow -lparquet -Xlinker -R,$(ENV_PATH)/lib

LOCAL_ARROW_I = -isystem $(LOCAL_ENV_PATH)/include
LOCAL_ARROW_L = -L $(LOCAL_ENV_PATH)/lib
LOCAL_ARROW_L_DYN = -larrow -lparquet -Xlinker -R,$(LOCAL_ENV_PATH)/lib

COADA_BASE = sbatch -p dgxa100 -t 12:00:00 --mem-per-cpu 64G
WATCH = watch -n 1 squeue -u vlad_adrian.ulmeanu
FILES = main.cu e3s_sawnoff.cu readers.cu utils.cu solvers/cses_solver.cu solvers/base_pq_solver.cu

build:
	nvcc $(FLAGS) main.cu utils.cu -o main

build_coada:
	$(COADA_BASE) --output=output_build_coada.log --wrap="time nvcc $(ARROW_I) $(ARROW_L) $(FLAGS) $(FILES) -o main $(ARROW_L_DYN)" && $(WATCH)

build_local:
	nvcc $(LOCAL_ARROW_I) $(LOCAL_ARROW_L) $(LOCAL_FLAGS) $(FILES) -o main $(LOCAL_ARROW_L_DYN)

build_local_debug:
	nvcc $(LOCAL_ARROW_I) $(LOCAL_ARROW_L) $(LOCAL_FLAGS) main.cu readers.cu utils.cu -o main $(LOCAL_ARROW_L_DYN)

build_test:
	nvcc $(FLAGS) $(FNAME).cu -o $(FNAME)

run_coada_cses:
	$(COADA_BASE) --output=output_run_coada.log --gres gpu:1 --wrap="time ./main cses $(INFILE) $(OUTFILE)" && $(WATCH)

run_coada_parquet:
	$(COADA_BASE) --output=output_run_coada.log --gres gpu:1 --wrap="time ./main $(TYPE_SOLVER) $(CNT_PARQUET_FILES) $(CNT_ATTACK_FILES) $(OUTFILE)" && $(WATCH)

run_local_cses:
	time ./main cses $(INFILE) $(OUTFILE) && diff -s $(OUTFILE) $(EXPFILE)

run_local_parquet:
	time ./main $(TYPE_SOLVER) $(CNT_PARQUET_FILES) $(CNT_ATTACK_FILES) $(OUTFILE)

# ! (local) conda activate cpp_libs, (server) cpp_libs2.

# FNAME = test_thrust / test_cub_segmented_scan
# INFILE = test_inputs/input0.txt
# EXPFILE = test_inputs/expected0.txt
# OUTFILE = test_inputs/output.txt
