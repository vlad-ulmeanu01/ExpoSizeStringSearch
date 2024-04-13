#include "E3S_utils.h"

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

    subOtp[0] = logOtp[0]; ///obligated to use the already generated value for length = 1.
    for (int j = 1; (1<<j) <= maxn; j++) {
        subOtp[j].first = otpDist(mt);
        subOtp[j].second = otpDist(mt);
    }
}

/**
 * We update the text on which we search the dictionary.
 * @param newS. The new string, in uint8_t format: newS[0 .. lengthNewS - 1].
 *              It already comes in lowered (case-nonsensitive) format.
 * @param lengthNewS. The length of the new string.
 */
void ExpoSizeStrSrc::updateText(const std::vector<uint8_t> &newS, int lengthNewS) {
    if (lengthNewS > curr_maxn) {
        curr_ml2 = 1 + int(log2(lengthNewS));

        while ((1<<strE2) < lengthNewS) {
            strE2++;
        }

        basePow.resize(lengthNewS + 1);
        if (curr_maxn == 0) {
            basePow[0] = std::make_pair(1, 1);
        }

        hhPref.resize(lengthNewS + 1);
        if (curr_maxn == 0) {
            hhPref[0] = std::make_pair(0, 0);
        }

        hash.resize((1 << curr_ml2) * curr_ml2);
        subtreeHash.resize((1 << curr_ml2) * curr_ml2);
        id.resize((1 << curr_ml2) * curr_ml2);
        leverage.resize((1 << curr_ml2) * curr_ml2);
        starterNodeChildren.resize(lengthNewS * curr_ml2);

        curr_maxn = lengthNewS;
    }

    n = lengthNewS;
    massSearchCntMatches = 0;
    cntStarterNodeChildren = 0;
    std::fill(massSearchResults.begin(), massSearchResults.end(), 0);

    int i, j, z;
    for (i = 1; i <= n; i++) {
        basePow[i] = basePow[i-1];
        mul2x(basePow[i].first, base.first, basePow[i].second, base.second);

        hhPref[i] = hhPref[i-1];
        mul2x(hhPref[i].first, base.first, hhPref[i].second, base.second);

        hhPref[i].first += (int)newS[i-1] + 1;
        hhPref[i].second += (int)newS[i-1] + 1;

        hhPref[i].first = (hhPref[i].first >= M61? hhPref[i].first - M61: hhPref[i].first);
        hhPref[i].second = (hhPref[i].second >= M61? hhPref[i].second - M61: hhPref[i].second);
    }

    ///compute the subtree hashes.
    std::pair<int64_t, int64_t> subtractedHash[256];

    int treeId = 0, subtreeHashSize = 0;
    for (j = 0; (1<<j) <= n; j++) {
        int len = std::min((1<<(j+1)) - 1, n);

        ///precalculating what we will decrease while rolling the hash.
        ///subtractedHash[z] = BASE ** (len - 1) * conv(z). conv(0) = 1.
        subtractedHash[0] = basePow[len - 1];
        for (z = 1; z < 256; z++) {
            subtractedHash[z].first = subtractedHash[z-1].first + basePow[len - 1].first;
            subtractedHash[z].second = subtractedHash[z-1].second + basePow[len - 1].second;

            subtractedHash[z].first = (subtractedHash[z].first >= M61? subtractedHash[z].first - M61: subtractedHash[z].first);
            subtractedHash[z].second = (subtractedHash[z].second >= M61? subtractedHash[z].second - M61: subtractedHash[z].second);
        }

        ///if the subtree hash would want more than we could possibly get from s, we will put just as much as we can.
        std::pair<int64_t, int64_t> hh(0, 0), tmp;
        for (i = 0; i < len; i++) {
            mul2x(hh.first, base.first, hh.second, base.second);
            hh.first += (int)newS[i] + 1;
            hh.second += (int)newS[i] + 1;

            hh.first = (hh.first >= M61? hh.first - M61: hh.first);
            hh.second = (hh.second >= M61? hh.second - M61: hh.second);
        }

        treeId = j * (1<<strE2);

        subtreeHash[subtreeHashSize++] = std::make_pair(reduceHash(hh, subOtp[j]), treeId);
        treeId++;

        for (i = 1; i + (1<<j) - 1 < n; i++) {
            if (i + len-1 < n) {
                hh.first -= subtractedHash[newS[i-1]].first;
                hh.second -= subtractedHash[newS[i-1]].second;

                hh.first = (hh.first < 0? hh.first + M61: hh.first);
                hh.second = (hh.second < 0? hh.second + M61: hh.second);

                mul2x(hh.first, base.first, hh.second, base.second);
                hh.first += (int)newS[i+len-1] + 1;
                hh.second += (int)newS[i+len-1] + 1;

                hh.first = (hh.first >= M61? hh.first - M61: hh.first);
                hh.second = (hh.second >= M61? hh.second - M61: hh.second);
            } else {
                tmp = std::pair<int64_t, int64_t>((int)newS[i-1] + 1, (int)newS[i-1] + 1);
                mul2x(tmp.first, basePow[n-i].first, tmp.second, basePow[n-i].second);
                hh.first -= tmp.first;
                hh.second -= tmp.second;

                hh.first = (hh.first < 0? hh.first + M61: hh.first);
                hh.second = (hh.second < 0? hh.second + M61: hh.second);
            }

            subtreeHash[subtreeHashSize++] = std::make_pair(reduceHash(hh, subOtp[j]), treeId);
            treeId++;
        }
    }

    ///sort all the subtree hashes. afterwards, we can compress the duplicates. (update the leverages, ids)
    radixSortPairs<std::pair<uint64_t, int>>(subtreeHash.begin(), 0, subtreeHashSize - 1);

    i = 0;
    while (i < subtreeHashSize) {
        j = i; ///go through all indexes with the same hash as subtreeHash[i].
        z = i; ///keep in z the index with the minimum id. also helpful when trying to solve problems like
        ///"find the first occurence of the words from ts in s"
        while (j < subtreeHashSize && subtreeHash[j].first == subtreeHash[i].first) {
            z = (subtreeHash[j].second < subtreeHash[z].second? j: z);
            j++;
        }

        ///unite all other nodes with the same hash in z.
        leverage[subtreeHash[z].second] = j - i;
        starterNodeChildren[cntStarterNodeChildren++] = subtreeHash[z].second;
        for (; i < j; i++) {
            id[subtreeHash[i].second] = subtreeHash[z].second;
        }
    }

    trieRoot->clear();
    trieRoot->idLevsCurrentlyHere.emplace_back(-1, INT_MAX); ///-1 is the starter node.
}

/**
 * How many times does a string appear in s?
 * @param t the string in question: t[0 .. lengthT - 1].
 *        t already comes in case non-sensitive format.
 * @param lengthT. The length of t.
 */
void ExpoSizeStrSrc::insertQueriedString(const std::vector<uint8_t> &t, int lengthT) {
    massSearchResults.push_back(0);
    if (lengthT > maxn || lengthT == 0) {
        return;
    }

    int i, j, z;
    TrieNode *trieNow = trieRoot, *trieNext = nullptr;
    std::pair<int64_t, int64_t> hh;
    uint64_t hh_red;

    ///split t into a substring chain, each substring having a distinct power of 2 length. add the chain to the trie.
    for (i = max_ml2, z = 0; i >= 0; i--) {
        if (lengthT & (1<<i)) {
            hh = std::make_pair(0, 0);
            for (j = z + (1<<i); z < j; z++) {
                mul2x(hh.first, base.first, hh.second, base.second);
                hh.first += (int)t[z] + 1;
                hh.second += (int)t[z] + 1;

                hh.first = (hh.first >= M61? hh.first - M61: hh.first);
                hh.second = (hh.second >= M61? hh.second - M61: hh.second);
            }

            hh_red = reduceHash(hh, logOtp[i]);

            auto it = trieNow->sons.find(hh_red);
            if (it != trieNow->sons.end()) {
                trieNow = it->second;
            } else {
                trieNext = trieNodeAlloc();
                trieNow->sons[hh_red] = trieNext;
                trieNow = trieNext;
            }
        }
    }

    trieNow->indexesEndingHere.push_back((int)massSearchResults.size() - 1);
}

/**
 * Recursively propagates what is in the given trie node.
 * @param trieNow current trie node to exploit.
 */
void ExpoSizeStrSrc::massSearch(TrieNode *trieNow) {
    if (!trieNow) return;

    int levSum = 0; ///compute the sum of leverages of all chains that are in the trie node.
    for (auto &x: trieNow->idLevsCurrentlyHere) {
        levSum += x.second;
    }

    if (!trieNow->indexesEndingHere.empty()) {
        massSearchCntMatches += levSum;
    }

    for (int x: trieNow->indexesEndingHere) { ///daca am mai multi indici, am mai multe string-uri duplicate in dictionar.
        massSearchResults[x] = levSum;
    }

    if (trieNow->sons.empty()) {
        return;
    }

    ///transform trieNow->sons in a sorted array. TODO sa faci asta in preparePatterns.
    int i = 0, cntSons = (int)trieNow->sons.size();
    std::pair<uint64_t, TrieNode *> sons[cntSons];
    for (auto &x: trieNow->sons) {
        sons[i++] = x;
    }

    int dagNode, nn, levChain, p2, startInd, dagNodeP2 = 0, dagNodeStartInd = 0;
    std::pair<int64_t, int64_t> hh;
    uint64_t hh_red;

    int m; ///how many children does the trie node have.
    for (auto &x: trieNow->idLevsCurrentlyHere) {
        std::tie(dagNode, levChain) = x;

        ///iterate through all of dagNode's children. compute the hashes of their nodes from the DAG. see if the hashes
        ///can be found on some edge that goes out from me (current trie node).

        ///count how many children does dagNode have.
        if (dagNode == -1) { ///am in the starter node.
            m = cntStarterNodeChildren;
        } else { ///decode from the dagNode index its length and starting index.
            dagNodeP2 = 1 << (dagNode >> strE2);
            dagNodeStartInd = dagNode - (dagNode >> strE2) * (1 << strE2);
            m = 0;
            while ((1<<m) < dagNodeP2 && dagNodeStartInd + dagNodeP2 + (1 << m) - 1 < n) {
                m++;
            }
        }

        for (i = 0; i < m; i++) {
            ///compute the hash of the i-th child of dagNode (hhl).
            ///also need the minimum value of the leverages along the current chain (levChain).
            if (dagNode == -1) {
                nn = starterNodeChildren[i];
                levChain = leverage[nn]; ///sthe lowest leverage on a chain is always the first. in this case it's leverage[nn].

                p2 = 1 << (nn >> strE2); ///length of the string associated to nn.
                startInd = nn - (nn >> strE2) * (1 << strE2); ///the index at which the associated string of nn begins in s.

                ///hh = hhPref[startInd + p2] - hhPref[startInd] * basePow[p2].
                hh = hhPref[startInd];
                mul2x(hh.first, basePow[p2].first, hh.second, basePow[p2].second);

                hh.first = hhPref[startInd + p2].first - hh.first;
                hh.second = hhPref[startInd + p2].second - hh.second;

                hh.first = (hh.first < 0? hh.first + M61: hh.first);
                hh.second = (hh.second < 0? hh.second + M61: hh.second);

                hh_red = reduceHash(hh, logOtp[nn >> strE2]);
                hash[nn] = hh_red; ///keep the value. might use it later in the search.
            } else {
                nn = id[(1 << strE2) * i + dagNodeStartInd + dagNodeP2]; ///we're sure that id[dagNode] = dagNode. it doesn't necessarily mean that id[nn] = nn in this case.
                hh_red = hash[nn]; ///hash was already precalculated when extending from the starter node.
            }

            auto it = std::lower_bound(sons, sons + cntSons, std::make_pair(hh_red, nullptr),
                                       [](const std::pair<uint64_t, TrieNode *> &a, std::pair<uint64_t, TrieNode *> b) {
                                           return a.first < b.first;
                                       });

            if (it != sons + cntSons && it->first == hh_red) {
                if (it->second->idLevsCurrentlyHere.empty() || it->second->idLevsCurrentlyHere.back().first != nn) {
                    it->second->idLevsCurrentlyHere.emplace_back(nn, levChain);
                } else {
                    ///duplicates may exist. possible to have multiple descendants with the same index after compression.
                    ///because of the mode in which we compress (merge in the lowest index) =>
                    ///we currently deal with a duplicate <=> nn == to the last id in it->second->idLevsCurrentlyHere.
                    it->second->idLevsCurrentlyHere.back().second += levChain;
                }
            }
        }
    }

    for (auto &x: trieNow->sons) {
        if (!x.second->idLevsCurrentlyHere.empty()) {
            massSearch(x.second);
        }
    }
}

void ExpoSizeStrSrc::trieBuffersFree() {
    for (TrieNode *tnBuff: TNBuffers) {
        delete[] tnBuff;
    }
}