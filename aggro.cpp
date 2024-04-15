///PTM HAI
#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math,O3")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native")

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <climits>
#include <cassert>
#include <chrono>
#include <vector>
#include <random>
#include <bitset>
#include <stack>
#include <cmath>
#include <map>

/**
 * @tparam T Sorts an array of pairs. The first element in the pairs has to be of type int/long long/__int128.
 * @param v Pointer to an array.
 * @param l First position that is sorted (v + l).
 * @param r Last position that is sorted (v + r).
 * the sort is unstable.
 */
template<typename T>
void radixSortPairs(T *v, int l, int r) {
    const int base = 256;

    std::array<std::vector<T>, 2> u;
    u[0].resize(r+1); u[1].resize(r+1);
    int cnt[base] = {0};

    int i, j, z, pin;

    auto mel = std::min_element(v+l, v+r+1)->first;
    if (mel > 0) mel = 0;

    for (i = l; i <= r; i++) {
        u[0][i].first = v[i].first - mel;
        u[0][i].second = v[i].second;
    }

    int noPasses = sizeof(v[l].first); ///4 for int, 8 for ll, 16 for __int128.
    for (i = 0, pin = 0; i < noPasses; i++, pin ^= 1) {
        std::fill(cnt, cnt + base, 0);

        for (j = l; j <= r; j++) {
            cnt[(u[pin][j].first >> (i << 3)) & 255]++;
        }

        for (j = 1; j < base; j++) {
            cnt[j] += cnt[j-1];
        }

        for (j = r; j >= l; j--) {
            z = ((u[pin][j].first >> (i << 3)) & 255);
            u[pin^1][l + (--cnt[z])] = u[pin][j];
        }
    }

    for (i = l; i <= r; i++) {
        v[i].first = u[pin][i].first + mel;
        v[i].second = u[pin][i].second;
    }
}

namespace xoshiro256pp {
    ///Written in 2019 by David Blackman and Sebastiano Vigna (vigna@acm.org)
    ///https://prng.di.unimi.it/splitmix64.c
    ///https://prng.di.unimi.it/xoshiro256plusplus.c
    ///https://vigna.di.unimi.it/ftp/papers/xorshift.pdf

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    ///using splitmix64 to initialize the state of xoshiro256++. we use it for only one value after each re-seeding.
    inline uint64_t seed_and_get(uint64_t hh1, uint64_t hh2) {
        hh1 += 0x9e3779b97f4a7c15; ///s[0] from xoshiro256plusplus.
        hh1 = (hh1 ^ (hh1 >> 30)) * 0xbf58476d1ce4e5b9;
        hh1 = (hh1 ^ (hh1 >> 27)) * 0x94d049bb133111eb;
        hh1 = hh1 ^ (hh1 >> 31);

        hh2 += 0x3c6ef372fe94f82a; ///s[3] from xoshiro256plusplus. 0x9e3779b97f4a7c15 * 2 - 2**64.
        hh2 = (hh2 ^ (hh2 >> 30)) * 0xbf58476d1ce4e5b9;
        hh2 = (hh2 ^ (hh2 >> 27)) * 0x94d049bb133111eb;
        hh2 = hh2 ^ (hh2 >> 31);

        return rotl(hh1 + hh2, 23) + hh1;
    }
}

class ExpoSizeStrSrc {
private:
    static constexpr uint32_t ct229 = (1 << 29) - 1;
    static constexpr uint64_t M61 = (1ULL << 61) - 1, M61_2x = M61 * 2;
    static constexpr int maxn = 100'000, ml2 = 17;

    std::pair<int64_t, int64_t> base, basePow[maxn+1]; ///the randomly chosen bases and their powers.
    std::pair<uint64_t, uint64_t> logOtp[ml2]; ///keep the one time pads for subsequences of lengths 1, 2, 4, ...

    int cntUncompressedNodes = 0;
    int64_t hash[(1<<ml2)*ml2]; ///effectively the hashes from the DAG nodes.
    std::pair<uint64_t, int> sortedHashes[(1<<ml2)*ml2]; ///still the hashes from the DAG nodes. first = hash, second = id.
    int id[(1<<ml2)*ml2]; ///in whom was a node united.

    int compressedGraph[maxn*ml2*ml2]; ///continous lists of [ids of the children in the compressed graph].
    std::pair<int, int> compressedGraphId[(1<<ml2)*ml2]; ///only if id[x] == x: [first, last) indexes for which we can look in compressedGraph for children ids.

    int n, strE2 = 0; ///2^strE2 is the smallest power of 2 that is >= n.
    std::string s;

    ///a1 = a1 * b1 % M61.
    ///a2 = a2 * b2 % M61.
    static void mul2x(int64_t &a1, const int64_t &b1, int64_t &a2, const int64_t &b2) {
        uint64_t a1_hi = a1 >> 32, a1_lo = (uint32_t)a1, b1_hi = b1 >> 32, b1_lo = (uint32_t)b1,
                a2_hi = a2 >> 32, a2_lo = (uint32_t)a2, b2_hi = b2 >> 32, b2_lo = (uint32_t)b2,
                ans_1 = 0, ans_2 = 0, tmp_1 = 0, tmp_2 = 0;

        tmp_1 = a1_hi * b1_lo + a1_lo * b1_hi;
        tmp_2 = a2_hi * b2_lo + a2_lo * b2_hi;

        tmp_1 = ((tmp_1 & ct229) << 32) + (tmp_1 >> 29);
        tmp_2 = ((tmp_2 & ct229) << 32) + (tmp_2 >> 29);

        tmp_1 += (a1_hi * b1_hi) << 3;
        tmp_2 += (a2_hi * b2_hi) << 3;

        ans_1 = (tmp_1 >> 61) + (tmp_1 & M61);
        ans_2 = (tmp_2 >> 61) + (tmp_2 & M61);

        tmp_1 = a1_lo * b1_lo;
        tmp_2 = a2_lo * b2_lo;

        ans_1 += (tmp_1 >> 61) + (tmp_1 & M61);
        ans_2 += (tmp_2 >> 61) + (tmp_2 & M61);

        ans_1 = (ans_1 >= M61_2x? ans_1 - M61_2x: (ans_1 >= M61? ans_1 - M61: ans_1));
        ans_2 = (ans_2 >= M61_2x? ans_2 - M61_2x: (ans_2 >= M61? ans_2 - M61: ans_2));

        a1 = ans_1;
        a2 = ans_2;
    }

    ///128bit hash ---(xorshift)---> uniform spread that fits in 8 bytes.
    static uint64_t reduceHash(std::pair<int64_t, int64_t> &hh, std::pair<uint64_t, uint64_t> otp) {
        otp.first ^= hh.first;
        otp.second ^= hh.second;

        return xoshiro256pp::seed_and_get(otp.first, otp.second);
    }

public:
    ExpoSizeStrSrc(std::string &s_) {
        s = std::move(s_);
        n = (int)s.size();

        int i, j, z;

        ///current time, current clock cycle count, heap address given by the OS. https://codeforces.com/blog/entry/60442
        std::seed_seq seq {
                (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
                (uint64_t) __builtin_ia32_rdtsc(),
                (uint64_t) (uintptr_t) std::make_unique<char>().get()
        };

        std::mt19937_64 mt(seq);
        std::uniform_int_distribution<int64_t> baseDist(27, M61 - 1);
        std::uniform_int_distribution<uint64_t> otpDist(0, ULLONG_MAX);
        base = std::make_pair(baseDist(mt), baseDist(mt)); ///uniformly and randomly choose 2 bases to use.

        while (base.second == base.first) {
            base.second = baseDist(mt);
        }

        ///compute logOtp for lengths 1, 2, 4, ...
        for (j = 0; (1<<j) <= n; j++) {
            logOtp[j].first = otpDist(mt);
            logOtp[j].second = otpDist(mt);
        }

        ///compute base powers, strE2.
        basePow[0] = std::make_pair(1, 1);
        for (i = 1; i <= n; i++) {
            basePow[i] = basePow[i-1];
            mul2x(basePow[i].first, base.first, basePow[i].second, base.second);
        }

        while ((1<<strE2) < n) {
            strE2++;
        }

        ///compute the DAG node hashes.
        std::pair<int64_t, int64_t> subtractedHash[26];

        int treeId = 0;
        for (j = 0, cntUncompressedNodes = 0; (1<<j) <= n; j++) {
            int len = (1<<j);

            ///precalculating what we will decrease while rolling the hash.
            ///subtractedHash[z] = 27 ** (len - 1) * conv('a' + z). conv('a') = 1.
            subtractedHash[0] = basePow[len - 1];
            for (z = 1; z < 26; z++) {
                subtractedHash[z].first = subtractedHash[z-1].first + basePow[len - 1].first;
                subtractedHash[z].second = subtractedHash[z-1].second + basePow[len - 1].second;

                subtractedHash[z].first = (subtractedHash[z].first >= M61? subtractedHash[z].first - M61: subtractedHash[z].first);
                subtractedHash[z].second = (subtractedHash[z].second >= M61? subtractedHash[z].second - M61: subtractedHash[z].second);
            }

            std::pair<int64_t, int64_t> hh(0, 0);
            for (i = 0; i < len; i++) {
                mul2x(hh.first, base.first, hh.second, base.second);
                hh.first += s[i] - 'a' + 1;
                hh.second += s[i] - 'a' + 1;

                hh.first = (hh.first >= M61? hh.first - M61: hh.first);
                hh.second = (hh.second >= M61? hh.second - M61: hh.second);
            }

            treeId = j * (1<<strE2);
            hash[treeId] = reduceHash(hh, logOtp[j]);
            sortedHashes[cntUncompressedNodes++] = std::make_pair(hash[treeId], treeId);
            treeId++;

            for (i = 1; i + (1<<j) - 1 < n; i++) {
                hh.first -= subtractedHash[s[i-1] - 'a'].first;
                hh.second -= subtractedHash[s[i-1] - 'a'].second;

                hh.first = (hh.first < 0? hh.first + M61: hh.first);
                hh.second = (hh.second < 0? hh.second + M61: hh.second);

                mul2x(hh.first, base.first, hh.second, base.second);
                hh.first += s[i+len-1] - 'a' + 1;
                hh.second += s[i+len-1] - 'a' + 1;

                hh.first = (hh.first >= M61? hh.first - M61: hh.first);
                hh.second = (hh.second >= M61? hh.second - M61: hh.second);

                hash[treeId] = reduceHash(hh, logOtp[j]);
                sortedHashes[cntUncompressedNodes++] = std::make_pair(hash[treeId], treeId);
                treeId++;
            }
        }

        ///sort all the DAG hashes. afterwards, we can compress the duplicates.
        radixSortPairs<std::pair<uint64_t, int>>(sortedHashes, 0, cntUncompressedNodes - 1);

        i = 0;
        while (i < cntUncompressedNodes) {
            j = i; ///go through all indexes with the same hash as subtreeHash[i].
            z = i; ///keep in z the index with the minimum id.
            while (j < cntUncompressedNodes && sortedHashes[j].first == sortedHashes[i].first) {
                z = (sortedHashes[j].second < sortedHashes[z].second? j: z);
                j++;
            }

            ///unite all other nodes with the same hash in z.
            for (; i < j; i++) {
                id[sortedHashes[i].second] = sortedHashes[z].second;
            }
        }

        i = 0;
        int cgIndexStart = 0, cgIndexCurr = 0; ///compressedGraph indexes.
        std::bitset<(1<<ml2) * ml2> bset;

        while (i < cntUncompressedNodes) {
            if (i < cntUncompressedNodes - ml2 + 1 && sortedHashes[i+ml2-1].first == sortedHashes[i].first) {
                j = i;

                int lenI = 1 << (sortedHashes[i].second >> strE2);
                while (j < cntUncompressedNodes && sortedHashes[j].first == sortedHashes[i].first) {
                    ///will iterate through the children of id = sortedHashes[j].second.
                    int offsetJ = sortedHashes[j].second - (1 << strE2) * (sortedHashes[j].second >> strE2);
                    for (z = 0; (1<<z) < lenI && offsetJ+lenI + (1<<z)-1 < n; z++) {
                        int idChild = (1 << strE2) * z + offsetJ+lenI;
                        if (!bset[id[idChild]]) {
                            compressedGraph[cgIndexCurr++] = id[idChild];
                            bset[id[idChild]] = true;
                        }
                    }

                    j++;
                }

                bset.reset();
                std::sort(compressedGraph + cgIndexStart, compressedGraph + cgIndexCurr);
                compressedGraphId[id[sortedHashes[i].second]] = std::make_pair(cgIndexStart, cgIndexCurr);

                cgIndexStart = cgIndexCurr;
                i = j;
            } else {
                ///there are too little (< log2(n)) DAG nodes with the same value. post compression, there aren't enough children to warrant
                ///sorting + binary-searching through the children ids. we can check each of the DAG nodes for the child in O(1), so still O(log n)
                ///per chain progression.
                i++;
            }
        }
    }

    /**
     * Does a string appear in s (Y/N)?
     * @param t the string in question.
     */
    bool searchString(std::string t) {
        if (t.empty() || (int)t.size() > n) {
            return false;
        }

        int m = __builtin_popcount(t.size());
        int exponents[m];
        uint64_t t_hashes[m];

        int i, j, z, k = 0;
        std::pair<int64_t, int64_t> hh;

        ///split t into a substring chain, each substring having a distinct power of 2 length.
        for (i = ml2-1, z = 0; i >= 0; i--) {
            if (t.size() & (1<<i)) {
                hh = std::make_pair(0, 0);
                for (j = z + (1<<i); z < j; z++) {
                    mul2x(hh.first, base.first, hh.second, base.second);
                    hh.first += t[z] - 'a' + 1;
                    hh.second += t[z] - 'a' + 1;

                    hh.first = (hh.first >= M61? hh.first - M61: hh.first);
                    hh.second = (hh.second >= M61? hh.second - M61: hh.second);
                }

                exponents[k] = i;
                t_hashes[k++] = reduceHash(hh, logOtp[i]);
            }
        }

        ///t_hashes[0] must exist in the compressed graph.
        z = std::lower_bound(sortedHashes, sortedHashes + cntUncompressedNodes, std::pair(t_hashes[0], 0)) - sortedHashes;
        if (z >= cntUncompressedNodes || sortedHashes[z].first != t_hashes[0]) return false;

        int l, r;
        for (i = 0; i < m-1; i++) {
            ///check if that compressedGraph list has t_hashes[i+1].
            k = std::lower_bound(sortedHashes, sortedHashes + cntUncompressedNodes, std::pair(t_hashes[i+1], 0)) - sortedHashes;
            if (k >= cntUncompressedNodes || sortedHashes[k].first != t_hashes[i+1]) return false;

            ///we know that t_hashes[i] exists. check if it has a child with the hash t_hashes[i+1].
            std::tie(l, r) = compressedGraphId[id[sortedHashes[z].second]]; ///get the compressedGraph interval for t_hashes[i].

            if (l < r) {
                if (!std::binary_search(compressedGraph + l, compressedGraph + r, id[sortedHashes[k].second])) {
                    return false;
                }
            } else {
                ///there are few DAG nodes with a hash value of t_hashes[i] (< log2(n)). we didn't put them in compressedGraph. find them here.
                ///per DAG node, there is also only one child with the correct length (1 << exponents[i+1]).
                int onlyId = id[sortedHashes[z].second];
                bool found = false;

                while (z < cntUncompressedNodes && !found && id[sortedHashes[z].second] == onlyId) {
                    int len = 1 << (sortedHashes[z].second >> strE2), offset = sortedHashes[z].second - (1 << strE2) * (sortedHashes[z].second >> strE2);

                    found |= (offset+len + (1 << exponents[i+1]) - 1 < n && id[(1 << strE2) * exponents[i+1] + offset+len] == id[sortedHashes[k].second]);
                    z++;
                }

                if (!found) {
                    return false;
                }
            }

            z = std::lower_bound(sortedHashes, sortedHashes + cntUncompressedNodes, std::pair(t_hashes[i+1], 0)) - sortedHashes;
        }

        return true;
    }
};

int main() {
    std::ios::sync_with_stdio(false); std::cin.tie(nullptr); std::cout.tie(nullptr);

    std::string s; std::cin >> s;

    ExpoSizeStrSrc *E3S = new ExpoSizeStrSrc(s);

    int n; std::cin >> n;
    std::string t;
    for (int i = 0; i < n; i++) {
        std::cin >> t;
        std::cout << ((E3S->searchString(t) && (t.size() <= 1 || E3S->searchString(t.substr(1))))? "YES\n": "NO\n");
    }

    delete E3S;

    return 0;
}
