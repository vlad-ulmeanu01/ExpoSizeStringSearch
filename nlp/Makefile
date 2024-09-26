
CXXFLAGS = -std=c++17 -O2

all: ./executables ./executables/myOffline ./executables/ahoCorasick_neal ./executables/suffixArray_ILHY ./executables/suffixAutomaton_randmod3 ./executables/suffixTree_elaina

./executables:
	mkdir -p ./executables

./executables/myOffline: sources/myOffline.cpp
	g++ $(CXXFLAGS) sources/myOffline.cpp -o executables/myOffline

./executables/ahoCorasick_neal: sources/ahoCorasick_neal.cpp
	g++ $(CXXFLAGS) sources/ahoCorasick_neal.cpp -o executables/ahoCorasick_neal

./executables/suffixArray_ILHY: sources/suffixArray_ILHY.cpp
	g++ $(CXXFLAGS) sources/suffixArray_ILHY.cpp -o executables/suffixArray_ILHY

./executables/suffixAutomaton_randmod3: sources/suffixAutomaton_randmod3.cpp
	g++ $(CXXFLAGS) sources/suffixAutomaton_randmod3.cpp -o executables/suffixAutomaton_randmod3

./executables/suffixTree_elaina: sources/suffixTree_elaina.cpp
	g++ $(CXXFLAGS) sources/suffixTree_elaina.cpp -o executables/suffixTree_elaina


clean:
	rm -f ./executables/*