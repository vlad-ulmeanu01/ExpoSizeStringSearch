build:
	g++ -std=c++17 -O2 -Wall \
		-I /export/home/acs/stud/v/vlad_adrian.ulmeanu/arrow_installed/include \
		-L /export/home/acs/stud/v/vlad_adrian.ulmeanu/arrow_installed/lib64 \
		-Wl,-R /export/home/acs/stud/v/vlad_adrian.ulmeanu/arrow_installed/lib64 \
		-larrow -lparquet -lthrift -lssl -lcrypto -lsnappy \
		main.cpp -o main
