///PTM HAI
#ifndef SNORT3_EXTRA_E3Saggrocl_UTILS_H
#define SNORT3_EXTRA_E3Saggrocl_UTILS_H

#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <climits>
#include <cassert>
#include <cctype>
#include <chrono>
#include <vector>
#include <random>
#include <bitset>
#include <stack>
#include <cmath>
#include <map>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

template<typename T>
void radixSortPairs(typename std::vector<T>::iterator v, int l, int r);

struct SharedInfo {
    static constexpr uint64_t M61 = (1ULL << 61) - 1;
    static constexpr int maxn = 65'535, max_ml2 = 16;
    
    std::pair<uint64_t, uint64_t> base; ///the randomly chosen bases.
    std::pair<uint64_t, uint64_t> basePow[max_ml2]; ///the bases' powers.
    std::pair<uint64_t, uint64_t> logOtp[max_ml2]; ///keep the one time pads for subsequences of lengths 1, 2, 4, ...

    static constexpr int profiler_debug_every = 10'000;
    std::chrono::steady_clock::time_point profiler_start;
    double profiler_timeSpent_updateText = 0, profiler_timeSpent_updateText_part1 = 0, profiler_timeSpent_massSearch = 0;
    int profiler_sort_total_size = 0, profiler_sort_times_called = 0, profiler_sort_max_size = 0;
    int profiler_counter = 0;

    SharedInfo();

    void debug_profiler();
};

///structura despre ce caut in DAG. caut o legatura hh1 -> hh2. nodul asociat lui hh2 are un string de lungime endExponent.
struct LinkInfo {
    std::pair<uint64_t, uint64_t> hashes;
    int endExponent;
    bool found;

    LinkInfo(): hashes(0, 0), endExponent(0), found(false) {};
    LinkInfo(uint64_t hh1, uint64_t hh2): hashes(hh1, hh2), endExponent(0), found(false) {}
    LinkInfo(uint64_t hh1, uint64_t hh2, int ee): hashes(hh1, hh2), endExponent(ee), found(false) {}

    bool operator < (const LinkInfo &oth) const { return hashes < oth.hashes; }
    bool operator == (const LinkInfo &oth) const { return hashes == oth.hashes; }
    bool operator != (const LinkInfo &oth) const { return hashes != oth.hashes; }
};

struct ChainInfo {
private:
    static constexpr int max_ml2 = 16;
public:
    ///hash chain info. initialized in ExpoSizeStrSrc::preprocessQueriedString. used in ExpoSizeStrSrc::queryString.
    uint64_t fullHash;
    int chainLength;
    std::array<int, max_ml2> exponents;
    std::array<uint64_t, max_ml2> t_hashes;
    std::array<int, max_ml2 - 1> massSearchIds; ///..[i] = where to search if t_h[i] -> t_h[i+1] exists.
};

///only a copy of ExpoSizeStrSrc must exist!
class ExpoSizeStrSrc {
private:
    static constexpr int maxn = 65'535, max_ml2 = 16;
    int curr_maxn = 0, curr_ml2 = 0;

    ///TODO: std::array.
    int cntUncompressedNodes;
    std::vector<uint64_t> hash; ///effectively the hashes from the DAG nodes. [(1<<ml2)*ml2]
    std::vector<std::pair<uint64_t, int>> sortedHashes; ///still the hashes from the DAG nodes. first = hash, second = id. [(1<<ml2)*ml2]
    std::vector<int> id; ///in whom was a node united. [(1<<ml2)*ml2]

    std::vector<int> compressedGraph; ///continous lists of [ids of the children in the compressed graph]. [maxn*ml2*ml2]
    std::vector<std::pair<int, int>> compressedGraphId; ///only if id[x] == x: [first, last) indexes for which we can look in compressedGraph for children ids. [(1<<ml2)*ml2]

    std::bitset<(1<<max_ml2) * max_ml2> updBset; ///used in updateText to mark duplicates when iterating through DAG node children.

    std::vector<uint64_t> hh_red_h; ///used only to get hh_red_d[] back.

    cl::Platform default_platform;
    cl::Device default_device;
    cl::Context context;
    cl::Program::Sources sources;
    cl::Program program;
    cl::CommandQueue queue;

    cl::Buffer new_s_d, pref_d, spad_d, hh_red_d, b_powers_d, otp_d;

    SharedInfo *sharedInfo;

    int strE2 = 0; ///2^strE2 is the smallest power of 2 that is >= n.
    int n;

public:
    ExpoSizeStrSrc(SharedInfo *sharedInfo);

    void updateText(const std::vector<uint8_t> &newS, int lengthNewS, std::vector<LinkInfo> &connections);

    void massSearch(std::vector<LinkInfo> &connections); ///search the [LinkInfo] dictionary against the current packet.
};

#endif //SNORT3_EXTRA_E3Saggrocl_UTILS_H
