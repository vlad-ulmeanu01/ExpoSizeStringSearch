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

	std::pair<int64_t, int64_t> base, basePow[maxn+1], hhPref[maxn+1]; ///the randomly chosen bases, their powers, and the hash prefixes of s.
	std::pair<uint64_t, uint64_t> logOtp[ml2]; ///keep the one time pads for subsequences of lengths 1, 2, 4, ...
	uint64_t hash[(1<<ml2)*ml2]; ///effectively the hashes from the DAG nodes. lazily calculated in massSearch (computed when propagating from the starter node, used later). call as hash[id].
	std::pair<uint64_t, int> subtreeHash[(1<<ml2)*ml2]; ///first = hash, second = id.
	int id[(1<<ml2)*ml2]; ///in whom was a node united.
	int cntStarterNodeChildren = 0, starterNodeChildren[maxn*ml2]; ///post compression, who are the starter node's children?
	std::pair<uint64_t, int> sonsHashes[maxn*ml2]; ///first = hash, second = id. used by the massSearch function.

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
	int cntDistinctSubstrings[maxn+1]; ///results after mass-search. after the mass-search this has to be updated like cnt[i] += cnt[i-1] forall i > 0.

	int massSearchArray[1+maxn+maxn*ml2]; ///since I can only propagate n + nlog2n tokens, I'd keep them whole in an array
	///rather than having vector<vi>s in the massSearch.
	int massSearchIndex; ///the first unused index in the massSearchArray.

	ExpoSizeStrSrc(std::string &s_) {
		s = s_;
		n = (int)s.size();

		std::fill(std::begin(cntDistinctSubstrings), std::end(cntDistinctSubstrings), 0);
		massSearchIndex = 0;

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

		///logOtp is not used in the constructor. here we need otps for lengths 1, 3, 7, ...
		///1, 2, 4, ... are used in queries.
		for (j = 0; (1<<j) <= n; j++) {
			logOtp[j].first = otpDist(mt);
			logOtp[j].second = otpDist(mt);
		}

		///compute base powers, hashes for the prefixes, strE2.
		basePow[0] = std::make_pair(1, 1);
		hhPref[0] = std::make_pair(0, 0);
		for (i = 1; i <= n; i++) {
			basePow[i] = basePow[i-1];
			mul2x(basePow[i].first, base.first, basePow[i].second, base.second);

			hhPref[i] = hhPref[i-1];
			mul2x(hhPref[i].first, base.first, hhPref[i].second, base.second);

			hhPref[i].first += s[i-1] - 'a' + 1;
			hhPref[i].second += s[i-1] - 'a' + 1;

			hhPref[i].first = (hhPref[i].first >= M61? hhPref[i].first - M61: hhPref[i].first);
			hhPref[i].second = (hhPref[i].second >= M61? hhPref[i].second - M61: hhPref[i].second);
		}

		while ((1<<strE2) < n) {
			strE2++;
		}

		///compute the subtree hashes.
		std::pair<int64_t, int64_t> subtractedHash[26];
		std::pair<uint64_t, uint64_t> otp; ///the subtrees have associated substring lengths of 1, 3, 7, ...

		int treeId = 0, subtreeHashSize = 0;
		for (j = 0; (1<<j) <= n; j++) {
			if (j == 0) {
				otp = logOtp[0]; ///obligated to use the already generated value for length = 1.
			} else {
				otp.first = otpDist(mt);
				otp.second = otpDist(mt);
			}

			int len = std::min((1<<(j+1)) - 1, n);

			///precalculating what we will decrease while rolling the hash.
			///subtractedHash[z] = 27 ** (len - 1) * conv('a' + z). conv('a') = 1.
			subtractedHash[0] = basePow[len - 1];
			for (z = 1; z < 26; z++) {
				subtractedHash[z].first = subtractedHash[z-1].first + basePow[len - 1].first;
				subtractedHash[z].second = subtractedHash[z-1].second + basePow[len - 1].second;

				subtractedHash[z].first = (subtractedHash[z].first >= M61? subtractedHash[z].first - M61: subtractedHash[z].first);
				subtractedHash[z].second = (subtractedHash[z].second >= M61? subtractedHash[z].second - M61: subtractedHash[z].second);
			}

			///if the subtree hash would want more than we could possibly get from s, we will put just as much as we can.
			std::pair<int64_t, int64_t> hh(0, 0), tmp;
			for (i = 0; i < len; i++) {
				mul2x(hh.first, base.first, hh.second, base.second);
				hh.first += s[i] - 'a' + 1;
				hh.second += s[i] - 'a' + 1;

				hh.first = (hh.first >= M61? hh.first - M61: hh.first);
				hh.second = (hh.second >= M61? hh.second - M61: hh.second);
			}

			treeId = j * (1<<strE2);

			subtreeHash[subtreeHashSize++] = std::make_pair(reduceHash(hh, otp), treeId);
			treeId++;

			for (i = 1; i + (1<<j) - 1 < n; i++) {
				if (i + len-1 < n) {
					hh.first -= subtractedHash[s[i-1] - 'a'].first;
					hh.second -= subtractedHash[s[i-1] - 'a'].second;

					hh.first = (hh.first < 0? hh.first + M61: hh.first);
					hh.second = (hh.second < 0? hh.second + M61: hh.second);

					mul2x(hh.first, base.first, hh.second, base.second);
					hh.first += s[i+len-1] - 'a' + 1;
					hh.second += s[i+len-1] - 'a' + 1;

					hh.first = (hh.first >= M61? hh.first - M61: hh.first);
					hh.second = (hh.second >= M61? hh.second - M61: hh.second);
				} else {
					tmp = std::pair<int64_t, int64_t>(s[i-1] - 'a' + 1, s[i-1] - 'a' + 1);
					mul2x(tmp.first, basePow[n-i].first, tmp.second, basePow[n-i].second);
					hh.first -= tmp.first;
					hh.second -= tmp.second;

					hh.first = (hh.first < 0? hh.first + M61: hh.first);
					hh.second = (hh.second < 0? hh.second + M61: hh.second);
				}

				subtreeHash[subtreeHashSize++] = std::make_pair(reduceHash(hh, otp), treeId);
				treeId++;
			}
		}

		///sort all the subtree hashes. afterwards, we can compress the duplicates. (also update the ids)
		radixSortPairs<std::pair<uint64_t, int>>(subtreeHash, 0, subtreeHashSize-1);

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
			starterNodeChildren[cntStarterNodeChildren++] = subtreeHash[z].second;
			for (; i < j; i++) {
				id[subtreeHash[i].second] = subtreeHash[z].second;
			}
		}
	}

	/**
	 * We are in a trie node with only one token. We want to calculate its subtree's contribution to the substring
	 * distribution. We don't have to propagate anymore to do that. Unique prefix means that
	 * any continuation is also unique.
	 * @param currLen if the current trie node chain is (starter node) --s[0]--> node[0] --s[1]--> node[1] --> ... --node[-1]--> s[-1],
	 * then currLen is len(s[0]) + len(s[1]) + .. + len(s[-1]). implicitly, if the node is the starter, currLen = 0.
	 * @param x id of the token. we guarantee that id[x] = x.
	 */
	void updTokenSuccesorsContribution(int currLen, int x) {
		int lengthX = 1 << (x >> strE2); ///the length of x's node.
		int indexEndX = x - (x >> strE2) * (1 << strE2) + lengthX - 1; ///the last index that is part of x. count from 0.

		///we have a new distinct substring for each length starting from currLen+1 (because we index from 0),
		///up to currLen+1 + <maximum number of characters that we can add while being in x's DAG subtree>.
		cntDistinctSubstrings[currLen + 1]++;
		cntDistinctSubstrings[currLen + 1 + std::min(lengthX - 1, n-1 - indexEndX)]--;
	}

	/**
	 * Recursively propagates what is in the given trie node.
	 * @param l index >= 0. used in the massSearchArray.
	 * @param r index > r.
	 * the current trie node is implicitly understood to contain the tokens (ids) found in massSearchArray[l..r).
	 * @param currLen if the current trie node chain is (starter node) --s[0]--> node[0] --s[1]--> node[1] --> ...
	 * --node[-1]--> s[-1], then currLen is len(s[0]) + len(s[1]) + .. + len(s[-1]).
	 * implicitly, if the node is the starter, currLen = 0.
	 */
	void massSearch(int l, int r, int currLen) {
		if (l == r) {
			return;
		}

		///all of the current node tokens show the same value of a subtring, just in different areas of the string.
		if (l > 0) { ///the trie root is a special case.
			///add one to count the trie node effectively.
			cntDistinctSubstrings[currLen]++;
			cntDistinctSubstrings[currLen+1]--;

			///if we only have one token showing the value of a substring, we have a distinct prefix that can't be found
			///anywhere else in the string, so we can just add all of its possible descendants now and not propagate
			///anymore.
			if (r - l == 1) {
				updTokenSuccesorsContribution(currLen, massSearchArray[l]);
				return;
			}
		}

		int i, j, z, m, nn, p2, dagNode, startInd, cntSons = 0, dagNodeP2 = 0, dagNodeStartInd = 0;
		std::pair<int64_t, int64_t> hh;
		uint64_t hh_red;

		///for each dagNode (token) in the trie, iterate through all of its DAG children and remember them.
		///eliminate all duplicates. for each distinct value of a DAG child, we have a new trie child.
		///ex we are in [aaaa]. s = ...aaaabcd...aaaabce...aaaabcd...aaaaddd...; we will have 2 trie children,
		///[bc] with 2 tokens, and [dd] with one token. [bc] has only 2 tokens because the third one is a duplicate.

		for (z = l; z < r; z++) {
			dagNode = massSearchArray[z];

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
				///compute the hash of the i-th child of dagNode (hh_red).
				if (dagNode == -1) {
					nn = starterNodeChildren[i];

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

				sonsHashes[cntSons++] = std::make_pair(hh_red, nn);
			}
		}

		//radixSortPairs<std::pair<uint64_t, int>>(sonsHashes, 0, cntSons-1);
		std::sort(sonsHashes, sonsHashes + cntSons);

		///sonsHashes looks like: [(hh_red1, nn1), (hh_red1, nn2), .., (hh_red1, nn3), ...]
		std::vector<std::pair<int, int>> sonLRs; ///keep <l, r> for each child.

		i = 0;
		while (i < cntSons) {
			l = r = massSearchIndex;

			j = i;
			while (j < cntSons && sonsHashes[j].first == sonsHashes[i].first) {
				if (l == r || massSearchArray[r-1] != sonsHashes[j].second) {
					massSearchArray[r++] = sonsHashes[j].second;
				}

				j++;
			}

			i = j;
			if (l+1 == r) {
				///won't waste space in the array. will compute now the contribution and not propagate. even if the number of tokens that
				///will propagate is < n + nlog2n, the number of tokens that may exist is O(nlog^2n). we don't want to
				///allocate that much extra memory.

				currLen += (1 << (massSearchArray[l] >> strE2));
				cntDistinctSubstrings[currLen]++;
				cntDistinctSubstrings[currLen+1]--;
				updTokenSuccesorsContribution(currLen, massSearchArray[l]);
				currLen -= (1 << (massSearchArray[l] >> strE2));

				massSearchIndex = l;
			} else {
				sonLRs.emplace_back(l, r);
				massSearchIndex = r;
			}
		}

		for (std::pair<int, int> &x: sonLRs) {
			massSearch(x.first, x.second, currLen + (1 << (massSearchArray[x.first] >> strE2)));
		}
	}
};

int main() {
	std::ios::sync_with_stdio(false); std::cin.tie(nullptr); std::cout.tie(nullptr);

	std::string s; std::cin >> s;

	ExpoSizeStrSrc *E3S = new ExpoSizeStrSrc(s);

	E3S->massSearchArray[E3S->massSearchIndex++] = -1;
	E3S->massSearch(0, 1, 0);

	for (int i = 1; i <= (int)s.size(); i++) {
		E3S->cntDistinctSubstrings[i] += E3S->cntDistinctSubstrings[i-1];
	}

	for (int i = 1; i <= (int)s.size(); i++) {
		std::cout << E3S->cntDistinctSubstrings[i] << ' ';
	}

	std::cout << '\n';

	delete E3S;

	return 0;
}

