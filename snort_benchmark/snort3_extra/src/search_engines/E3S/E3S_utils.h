///PTM HAI
#ifndef SNORT3_EXTRA_E3S_UTILS_H
#define SNORT3_EXTRA_E3S_UTILS_H

#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math,O3")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native")

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
private:
    static constexpr uint32_t ct229 = (1 << 29) - 1;
    static constexpr uint64_t M61 = (1ULL << 61) - 1, M61_2x = M61 * 2;
    static constexpr int maxn = 65'535, max_ml2 = 16;
    int curr_maxn = 0, curr_ml2 = 0;

    ///trie built from dictionary entries.
    struct TrieNode {
        std::vector<int> indexesEndingHere; ///the indexes whose dictionary strings end here.

        std::map<uint64_t, TrieNode *> sons; ///do I have a son with some associated hash?
        std::vector<std::pair<uint64_t, TrieNode *>> arrSons;

        std::vector<std::pair<int, int>> idLevsCurrentlyHere; ///keep track of tokens that are in this trie node.

        void clear() {
            idLevsCurrentlyHere.clear();
            for (auto &x: arrSons) {
                x.second->clear();
            }
        }
    };

    const int TNBufSz = 4096;
    int TNBufInd = TNBufSz;
    std::vector<TrieNode *> TNBuffers;

    TrieNode *trieNodeAlloc() {
        if (TNBufInd >= TNBufSz) {
            TNBuffers.push_back(new TrieNode[TNBufSz]);
            TNBufInd = 0;
        }

        return &TNBuffers.back()[TNBufInd++];
    }

    std::pair<int64_t, int64_t> base; ///the randomly chosen bases.
    std::pair<uint64_t, uint64_t> logOtp[max_ml2]; ///keep the one time pads for subsequences of lengths 1, 2, 4, ...
    std::pair<uint64_t, uint64_t> subOtp[max_ml2]; ///keep the one time pads for subsequences of lengths 1, 3, 7, 15, ... (for the subtrees)

    std::vector<std::pair<int64_t, int64_t>> basePow, ///the bases' powers [maxn+1].
                                             hhPref; ///the hash prefixes of s [maxn+1].

    std::vector<uint64_t> hash; ///effectively the hashes from the DAG nodes. lazily calculated in massSearch (computed when propagating from the
                                ///starter node, used later). call as hash[id]. [(1<<ml2)*ml2]

    std::vector<std::pair<uint64_t, int>> subtreeHash; ///first = hash, second = id. [(1<<ml2)*ml2]
    std::vector<int> id; ///in whom was a node united. [(1<<ml2)*ml2]

    std::vector<int> leverage; ///lev[x] = how many nodes were united in x. also consider x when counting. [(1<<ml2)*ml2]
    std::vector<int> starterNodeChildren; ///post compression, who are the starter node's children? [maxn*ml2]
    int cntStarterNodeChildren;

    int n;
    int strE2 = 0; ///2^strE2 is the smallest power of 2 that is >= n.

    ///a1 = a1 * b1 % M61.
    ///a2 = a2 * b2 % M61.
    static void mul2x(int64_t &a1, const int64_t &b1, int64_t &a2, const int64_t &b2);

    ///128bit hash ---(xorshift)---> uniform spread that fits in 8 bytes.
    static uint64_t reduceHash(std::pair<int64_t, int64_t> &hh, std::pair<uint64_t, uint64_t> otp);

public:
    TrieNode *trieRoot = trieNodeAlloc();
    std::vector<int> massSearchResults; ///results after mass-search. how many times does .. appear in s?
    int massSearchCntMatches;

    ExpoSizeStrSrc();

    void updateText(const std::vector<uint8_t> &newS, int lengthNewS);

    void insertQueriedString(const std::vector<uint8_t> &t, int lengthT);

    void massSearch(TrieNode *trieNow);

    void linearizeMaps(TrieNode *trieNow);

    void trieBuffersFree();
};

#endif //SNORT3_EXTRA_E3S_UTILS_H
