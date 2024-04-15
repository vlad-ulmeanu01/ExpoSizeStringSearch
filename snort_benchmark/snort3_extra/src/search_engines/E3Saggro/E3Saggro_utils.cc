#include "E3Saggro_utils.h"

/**
 * @tparam T Sorts an array of pairs. The first element in the pairs has to be of type int/long long/__int128.
 * @param v Pointer to an array.
 * @param l First position that is sorted (v + l).
 * @param r Last position that is sorted (v + r).
 * the sort is unstable.
 */
template<typename T>
void radixSortPairs(typename std::vector<T>::iterator v, int l, int r) {
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
    static inline uint64_t seed_and_get(uint64_t hh1, uint64_t hh2) {
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

///a1 = a1 * b1 % M61.
///a2 = a2 * b2 % M61.
void ExpoSizeStrSrc::mul2x(int64_t &a1, const int64_t &b1, int64_t &a2, const int64_t &b2) {
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
uint64_t ExpoSizeStrSrc::reduceHash(std::pair<int64_t, int64_t> &hh, std::pair<uint64_t, uint64_t> otp) {
    otp.first ^= hh.first;
    otp.second ^= hh.second;

    return xoshiro256pp::seed_and_get(otp.first, otp.second);
}

ExpoSizeStrSrc::ExpoSizeStrSrc() {
    ///current time, current clock cycle count, heap address given by the OS. https://codeforces.com/blog/entry/60442
    std::seed_seq seq {
            (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
            (uint64_t) __builtin_ia32_rdtsc(),
            (uint64_t) (uintptr_t) std::make_unique<char>().get()
    };

    std::mt19937_64 mt(seq);
    std::uniform_int_distribution<int64_t> baseDist(257, M61 - 1);
    std::uniform_int_distribution<uint64_t> otpDist(0, ULLONG_MAX);
    base = std::make_pair(baseDist(mt), baseDist(mt)); ///uniformly and randomly choose 2 bases to use.

    while (base.second == base.first) {
        base.second = baseDist(mt);
    }

    ///logOtp is not used in the constructor. here we need otps for lengths 1, 3, 7, ...
    ///1, 2, 4, ... are used in queries.
    for (int j = 0; (1<<j) <= maxn; j++) {
        logOtp[j].first = otpDist(mt);
        logOtp[j].second = otpDist(mt);
    }
}

/**
 * We update the text on which we search the dictionary.
 * @param newS. The new string, in uint8_t format: newS[0 .. lengthNewS - 1].
 *              It already comes in lowered (case-nonsensitive) format.
 * @param lengthNewS. The length of the new string.
 * @param connections. a sorted array [(hh1, hh2)]. we eventually have to find DAG links hh1 -> hh2.
 *                     we use it to skip over DAG links that we never have to query.
 */
void ExpoSizeStrSrc::updateText(const std::vector<uint8_t> &newS, int lengthNewS, std::vector<LinkInfo> &connections) {
    if (lengthNewS > curr_maxn) {
        curr_ml2 = 1 + int(log2(lengthNewS));

        while ((1<<strE2) < lengthNewS) {
            strE2++;
        }

        basePow.resize(lengthNewS + 1);
        if (curr_maxn == 0) {
            basePow[0] = std::make_pair(1, 1);
        }

        for (int i = curr_maxn + 1; i <= lengthNewS; i++) {
            basePow[i] = basePow[i-1];
            mul2x(basePow[i].first, base.first, basePow[i].second, base.second);
        }

        hash.resize((1 << curr_ml2) * curr_ml2);
        sortedHashes.resize((1 << curr_ml2) * curr_ml2);
        id.resize((1 << curr_ml2) * curr_ml2);
        compressedGraph.resize(lengthNewS * curr_ml2 * curr_ml2);

        compressedGraphId.resize((1 << curr_ml2) * curr_ml2);
        std::fill(compressedGraphId.begin(), compressedGraphId.end(), std::make_pair(0, 0));

        curr_maxn = lengthNewS;
    }

    n = lengthNewS;
    cntUncompressedNodes = 0;

    ///compute the DAG node hashes.
    std::pair<int64_t, int64_t> subtractedHash[256];

    int i, j, z, treeId = 0;
    for (j = 0, cntUncompressedNodes = 0; (1<<j) <= n; j++) {
        int len = (1<<j);

        ///precalculating what we will decrease while rolling the hash.
        ///subtractedHash[z] = BASE ** (len - 1) * conv(z). conv(0) = 1.
        subtractedHash[0] = basePow[len - 1];
        for (z = 1; z < 256; z++) {
            subtractedHash[z].first = subtractedHash[z-1].first + basePow[len - 1].first;
            subtractedHash[z].second = subtractedHash[z-1].second + basePow[len - 1].second;

            subtractedHash[z].first = (subtractedHash[z].first >= M61? subtractedHash[z].first - M61: subtractedHash[z].first);
            subtractedHash[z].second = (subtractedHash[z].second >= M61? subtractedHash[z].second - M61: subtractedHash[z].second);
        }

        std::pair<int64_t, int64_t> hh(0, 0);
        for (i = 0; i < len; i++) {
            mul2x(hh.first, base.first, hh.second, base.second);
            hh.first += (int)newS[i] + 1;
            hh.second += (int)newS[i] + 1;

            hh.first = (hh.first >= M61? hh.first - M61: hh.first);
            hh.second = (hh.second >= M61? hh.second - M61: hh.second);
        }

        treeId = j * (1<<strE2);
        hash[treeId] = reduceHash(hh, logOtp[j]);
        sortedHashes[cntUncompressedNodes++] = std::make_pair(hash[treeId], treeId);
        treeId++;

        for (i = 1; i + (1<<j) - 1 < n; i++) {
            hh.first -= subtractedHash[newS[i-1]].first;
            hh.second -= subtractedHash[newS[i-1]].second;

            hh.first = (hh.first < 0? hh.first + M61: hh.first);
            hh.second = (hh.second < 0? hh.second + M61: hh.second);

            mul2x(hh.first, base.first, hh.second, base.second);
            hh.first += (int)newS[i+len-1] + 1;
            hh.second += (int)newS[i+len-1] + 1;

            hh.first = (hh.first >= M61? hh.first - M61: hh.first);
            hh.second = (hh.second >= M61? hh.second - M61: hh.second);

            hash[treeId] = reduceHash(hh, logOtp[j]);
            sortedHashes[cntUncompressedNodes++] = std::make_pair(hash[treeId], treeId);
            treeId++;
        }
    }

    ///sort all the DAG hashes. afterwards, we can compress the duplicates.
    radixSortPairs<std::pair<uint64_t, int>>(sortedHashes.begin(), 0, cntUncompressedNodes - 1);

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
    int connIndex = 0; ///pointer for the (sorted) connections[] array.

    updBset.reset();
    while (i < cntUncompressedNodes) {
        while (connIndex < (int)connections.size() && connections[connIndex].hashes.first < sortedHashes[i].first) {
            connIndex++;
        }

        if (connIndex >= (int)connections.size() || connections[connIndex].hashes.first != sortedHashes[i].first) {
            ///there is no link in connections that begins with sortedHashes[i].first. we will never query anything related, skip.
            i++;
        } else if (i < cntUncompressedNodes - max_ml2 + 1 && sortedHashes[i+max_ml2-1].first == sortedHashes[i].first) {
            ///we have at least log2(n) DAG nodes with the same hash value. keep track of their children. when there are
            ///enough DAG nodes with the same value, we can assume that most of their children are similar. we use a bitset
            ///to mark duplicates instead of calling std::unique later.

            j = i;

            int lenI = 1 << (sortedHashes[i].second >> strE2);
            while (j < cntUncompressedNodes && sortedHashes[j].first == sortedHashes[i].first) {
                ///will iterate through the children of id = sortedHashes[j].second.
                int offsetJ = sortedHashes[j].second - (1 << strE2) * (sortedHashes[j].second >> strE2);
                for (z = 0; (1<<z) < lenI && offsetJ+lenI + (1<<z)-1 < n; z++) {
                    int idChild = (1 << strE2) * z + offsetJ+lenI;
                    if (!updBset[id[idChild]]) {
                        compressedGraph[cgIndexCurr++] = id[idChild];
                        updBset[id[idChild]] = true;
                    }
                }

                j++;
            }

            updBset.reset();
            std::sort(compressedGraph.begin() + cgIndexStart, compressedGraph.begin() + cgIndexCurr);
            compressedGraphId[id[sortedHashes[i].second]] = std::make_pair(cgIndexStart, cgIndexCurr);

            cgIndexStart = cgIndexCurr;
            i = j;
        } else {
            ///there are too little (< log2(n)) DAG nodes with the same value. post compression, there aren't enough children to warrant
            ///sorting + binary-searching through the children ids. we can check each of the DAG nodes for the child in O(1) (because
            /// we know the length of the child at which we want to look), so still O(log n) per chain progression.
            i++;
        }
    }
}

/**
 * aggro is inherently online. since we will search the same string over different texts, we should remember in the pre-search phase the hash chain.
 * @param t the string in question: t[0 .. lengthT - 1]. t already comes in case non-sensitive format.
 * @param lengthT. The length of t.
 * @param ci.fullHash. the entire hash of t.
 * @param ci.chainLength. must be filled in this function. the length of the hash chain.
 * @param ci.exponents. also must be filled here. the powers of two that make lengthT, in decreasing order. (i.e. for 11, [8, 2, 1])
 * @param ci.t_hashes. also must be filled here. the elements of the hash chain.
 */
void ExpoSizeStrSrc::preprocessQueriedString(const std::vector<uint8_t> &t, int lengthT, ChainInfo &ci) {
    ci.chainLength = __builtin_popcount(lengthT);

    ///split t into a substring chain, each substring having a distinct power of 2 length.

    int i, j, z, k = 0;
    std::pair<int64_t, int64_t> hh, fullHh128(0, 0);
    for (i = max_ml2, z = 0; i >= 0; i--) {
        if (lengthT & (1<<i)) {
            hh = std::make_pair(0, 0);
            for (j = z + (1<<i); z < j; z++) {
                mul2x(hh.first, base.first, hh.second, base.second);
                mul2x(fullHh128.first, base.first, fullHh128.second, base.second);

                hh.first += (int)t[z] + 1;
                hh.second += (int)t[z] + 1;

                fullHh128.first += (int)t[z] + 1;
                fullHh128.second += (int)t[z] + 1;

                hh.first = (hh.first >= M61? hh.first - M61: hh.first);
                hh.second = (hh.second >= M61? hh.second - M61: hh.second);

                fullHh128.first = (fullHh128.first >= M61? fullHh128.first - M61: fullHh128.first);
                fullHh128.second = (fullHh128.second >= M61? fullHh128.second - M61: fullHh128.second);
            }

            ci.exponents[k] = i;
            ci.t_hashes[k++] = reduceHash(hh, logOtp[i]);
        }
    }

    ci.fullHash = reduceHash(fullHh128, std::make_pair(0, 0));
}

/**
 * For each pair (hh1, hh2) in connections, does hh1 -> hh2 appear in the text's DAG? update connections[..].found with the answers.
 * Some queried strings may have a length that is exactly a power of two. Their chains have a length of 1 (so no links).
 * They still have entries in connections (looking like (hh1, hh1)).
 * @param connections
 */
void ExpoSizeStrSrc::massSearch(std::vector<LinkInfo> &connections) {
    int z, k, l, r;
    for (auto &conn: connections) {
        conn.found = false;

        if (conn.hashes.first == conn.hashes.second) {
            ///false link. corresponding queried string's length is exactly a power of two, so the chain's length is 1.
            ///we only check if conn.hashes is present in sortedHashes.

            z = std::lower_bound(sortedHashes.begin(), sortedHashes.begin() + cntUncompressedNodes, std::make_pair(conn.hashes.first, 0)) - sortedHashes.begin();
            conn.found = (z < cntUncompressedNodes && sortedHashes[z].first == conn.hashes.first);
        } else {
            ///check if conn.hashes.first exists.
            z = std::lower_bound(sortedHashes.begin(), sortedHashes.begin() + cntUncompressedNodes, std::make_pair(conn.hashes.first, 0)) - sortedHashes.begin();
            if (z >= cntUncompressedNodes || sortedHashes[z].first != conn.hashes.first) continue;

            ///check if conn.hashes.second exists.
            k = std::lower_bound(sortedHashes.begin(), sortedHashes.begin() + cntUncompressedNodes, std::make_pair(conn.hashes.second, 0)) - sortedHashes.begin();
            if (k >= cntUncompressedNodes || sortedHashes[k].first != conn.hashes.second) continue;

            std::tie(l, r) = compressedGraphId[id[sortedHashes[z].second]]; ///get the compressedGraph interval for conn.hashes.first.

            if (l < r) {
                ///we already collected (and sorted) the hashes of the children of anybody with the hash conn.hashes.first.
                if (!std::binary_search(compressedGraph.begin() + l, compressedGraph.begin() + r, id[sortedHashes[k].second])) {
                    continue;
                }
            } else {
                ///there are few DAG nodes with a hash value of t_hashes[i] (< log2(n)). we didn't put them in compressedGraph. find them here.
                ///per DAG node, there is also only one child with the correct length conn.endExponent.
                int onlyId = id[sortedHashes[z].second], len = 1 << (sortedHashes[z].second >> strE2), offset;
                bool found = false;

                while (z < cntUncompressedNodes && !found && id[sortedHashes[z].second] == onlyId) {
                    offset = sortedHashes[z].second - (1 << strE2) * (sortedHashes[z].second >> strE2);

                    found |= (offset+len + (1 << conn.endExponent) - 1 < n && id[(1 << strE2) * conn.endExponent + offset+len] == id[sortedHashes[k].second]);
                    z++;
                }

                if (!found) {
                    continue;
                }
            }

            conn.found = true;
        }
    }
}
