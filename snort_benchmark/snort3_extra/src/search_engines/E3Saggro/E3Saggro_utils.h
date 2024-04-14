///PTM HAI
#ifndef SNORT3_EXTRA_E3SAGGRO_UTILS_H
#define SNORT3_EXTRA_E3SAGGRO_UTILS_H

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

template<typename T>
void radixSortPairs(typename std::vector<T>::iterator v, int l, int r);

namespace xoshiro256pp {
    static inline uint64_t rotl(const uint64_t x, int k);
    static inline uint64_t seed_and_get(uint64_t hh1, uint64_t hh2);
}

class ExpoSizeStrSrc {
public:
    static constexpr int max_ml2 = 16;

private:
    static constexpr uint32_t ct229 = (1 << 29) - 1;
    static constexpr uint64_t M61 = (1ULL << 61) - 1, M61_2x = M61 * 2;
    static constexpr int maxn = 65'535;
    int curr_maxn = 0, curr_ml2 = 0;

    std::pair<int64_t, int64_t> base; ///the randomly chosen bases.
    std::vector<std::pair<int64_t, int64_t>> basePow; ///the bases' powers. [maxn+1]
    std::pair<uint64_t, uint64_t> logOtp[max_ml2]; ///keep the one time pads for subsequences of lengths 1, 2, 4, ...

    int cntUncompressedNodes;
    std::vector<uint64_t> hash; ///effectively the hashes from the DAG nodes. [(1<<ml2)*ml2]
    std::vector<std::pair<uint64_t, int>> sortedHashes; ///still the hashes from the DAG nodes. first = hash, second = id. [(1<<ml2)*ml2]
    std::vector<int> id; ///in whom was a node united. [(1<<ml2)*ml2]

    std::vector<int> compressedGraph; ///continous lists of [ids of the children in the compressed graph]. [maxn*ml2*ml2]
    std::vector<std::pair<int, int>> compressedGraphId; ///only if id[x] == x: [first, last) indexes for which we can look in compressedGraph for children ids. [(1<<ml2)*ml2]

    std::bitset<(1<<max_ml2) * max_ml2> updBset; ///used in updateText to mark duplicates when iterating through DAG node children.

    int strE2 = 0; ///2^strE2 is the smallest power of 2 that is >= n.
    int n;

    ///a1 = a1 * b1 % M61.
    ///a2 = a2 * b2 % M61.
    static void mul2x(int64_t &a1, const int64_t &b1, int64_t &a2, const int64_t &b2);

    ///128bit hash ---(xorshift)---> uniform spread that fits in 8 bytes.
    static uint64_t reduceHash(std::pair<int64_t, int64_t> &hh, std::pair<uint64_t, uint64_t> otp);

public:
    ExpoSizeStrSrc();

    void updateText(const std::vector<uint8_t> &newS, int lengthNewS);

    void preprocessQueriedString(const std::vector<uint8_t> &t, int lengthT,
                                 int &chainLength, std::array<int, max_ml2> &exponents, std::array<uint64_t, max_ml2> &t_hashes);

    bool queryString(const int &chainLength, const std::array<int, max_ml2> &exponents, const std::array<uint64_t, max_ml2> &t_hashes);
};

#endif //SNORT3_EXTRA_E3SAGGRO_UTILS_H
