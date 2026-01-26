build:
	g++ -std=c++17 -O2 -Wall \
		-I /export/home/acs/stud/v/vlad_adrian.ulmeanu/arrow_installed/include \
		-L /export/home/acs/stud/v/vlad_adrian.ulmeanu/arrow_installed/lib64 \
		-Wl,-R /export/home/acs/stud/v/vlad_adrian.ulmeanu/arrow_installed/lib64 \
		-larrow -lparquet -lthrift -lssl -lcrypto -lsnappy \
		main.cpp -o main

build_hopeful:
	g++ -std=c++17 -O2 -Wall -Wextra \
		-I /home/vlad/miniconda3/envs/cpp_libs/include \
		-L /home/vlad/miniconda3/envs/cpp_libs/lib \
		main.cpp -o main \
		-larrow -lparquet \
		-Wl,-R /home/vlad/miniconda3/envs/cpp_libs/lib

build_nvcc:
	nvcc -std=c++17 -O2 -arch=sm_61 \
		-I /home/vlad/miniconda3/envs/cpp_libs/include \
		-L /home/vlad/miniconda3/envs/cpp_libs/lib \
		main.cpp -o main \
		-larrow -lparquet \
		-Xlinker -R,/home/vlad/miniconda3/envs/cpp_libs/lib