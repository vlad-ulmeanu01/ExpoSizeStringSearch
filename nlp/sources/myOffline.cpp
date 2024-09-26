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
 
static constexpr uint32_t ct229 = (1 << 29) - 1;
static constexpr uint64_t M61 = (1ULL << 61) - 1, M61_2x = M61 * 2;
static constexpr int maxn = 120'000, ml2 = 17;

class ExpoSizeStrSrc {
private:
 
	// ///trie built from dictionary entries.
	// struct TrieNode {
	// 	std::vector<int> indexesEndingHere; ///the indexes whose dictionary strings end here.
	// 	std::map<uint64_t, TrieNode *> sons; ///do I have a son with some associated hash?
	// 	std::vector<std::pair<int, int>> idLevsCurrentlyHere; ///keep track of tokens that are in this trie node.
	// };
 
    struct TrieNode {
        std::vector<int> indexesEndingHere; ///the indexes whose dictionary strings end here.

        std::map<uint64_t, TrieNode *> sons; ///do I have a son with some associated hash?
        // std::vector<std::pair<uint64_t, TrieNode *>> arrSons;
        std::vector<std::pair<int, int>> idLevsCurrentlyHere; ///keep track of tokens that are in this trie node.

        void clear() {
            idLevsCurrentlyHere.clear();
            for (auto &x: sons) {
                x.second->clear();
            }
        }
    };

	const int TNBufSz = 4096;
	int TNBufInd = TNBufSz;
	TrieNode *TNBuffer = nullptr;
 
	TrieNode *trieNodeAlloc() {
		if (TNBufInd >= TNBufSz) {
			TNBuffer = new TrieNode[TNBufSz];
			TNBufInd = 0;
		}
 
		return &TNBuffer[TNBufInd++];
	}
 
	std::pair<int64_t, int64_t> base, basePow[maxn+1], hhPref[maxn+1]; ///the randomly chosen bases, their powers, and the hash prefixes of s.
	std::pair<uint64_t, uint64_t> logOtp[ml2]; ///keep the one time pads for subsequences of lengths 1, 2, 4, ...
	uint64_t hash[(1<<ml2)*ml2]; ///effectively the hashes from the DAG nodes. lazily calculated in massSearch (computed when propagating from the starter node, used later). call as hash[id].
	std::pair<uint64_t, int> subtreeHash[(1<<ml2)*ml2]; ///first = hash, second = id.
	int id[(1<<ml2)*ml2]; ///in whom was a node united.
	int leverage[(1<<ml2)*ml2]; ///lev[x] = how many nodes were united in x. also consider x when counting.
	int cntStarterNodeChildren = 0, starterNodeChildren[maxn*ml2]; ///post compression, who are the starter node's children?
 
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
	TrieNode *trieRoot = trieNodeAlloc();
	std::vector<int> massSearchResults; ///results after mass-search. how many times does .. appear in s?
 
	void setupDag(std::string &s_) {
		s = s_;
		n = (int)s.size();
		cntStarterNodeChildren = 0;

		int i, j, z;
 
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
	//				std::cerr << s[i - 1] << " " << i << "\n";
	//				std::cerr << s[i-1] - 'a' << "\n";
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

			// massSearchResults.clear();
		}
 
		///sort all the subtree hashes. afterwards, we can compress the duplicates. (update the leverages, ids)
		radixSortPairs<std::pair<uint64_t, int>>(subtreeHash, 0, subtreeHashSize-1);
 			
 		int counter = 0;
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
		trieRoot->idLevsCurrentlyHere.emplace_back(-1, INT_MAX);
	}

	std::mt19937_64 mt;
	std::uniform_int_distribution<int64_t> baseDist;
	std::uniform_int_distribution<uint64_t> otpDist;

	ExpoSizeStrSrc():baseDist(27, M61 - 1), otpDist(0, ULLONG_MAX) {

		///current time, current clock cycle count, heap address given by the OS. https://codeforces.com/blog/entry/60442
		std::seed_seq seq {
				(uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
				(uint64_t) __builtin_ia32_rdtsc(),
				(uint64_t) (uintptr_t) std::make_unique<char>().get()
		};
 
		mt = std::mt19937_64(seq);
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

 
		trieRoot->idLevsCurrentlyHere.emplace_back(-1, INT_MAX); ///-1 is the starter node.
	}
 
	/**
	 * How many times does a string appear in s?
	 * @param t the string in question.
	 */
	void insertQueriedString(std::string t) {

		if (t.size() > maxn || t.empty()) {
			return;
		}

		massSearchResults.push_back(0);
 
		int i, j, z;
		TrieNode *trieNow = trieRoot, *trieNext = nullptr;
		std::pair<int64_t, int64_t> hh;
		uint64_t hh_red;
 
		///split t into a substring chain, each substring having a distinct power of 2 length. add the chain to the trie.
		for (i = ml2, z = 0; i >= 0; i--) {
			if (t.size() & (1<<i)) {
				hh = std::make_pair(0, 0);
				for (j = z + (1<<i); z < j; z++) {
					mul2x(hh.first, base.first, hh.second, base.second);
					hh.first += t[z] - 'a' + 1;
					hh.second += t[z] - 'a' + 1;
 
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
	void massSearch(TrieNode *trieNow) {
		if (!trieNow) return;
 
		int levSum = 0; ///compute the sum of leverages of all chains that are in the trie node.
		for (auto &x: trieNow->idLevsCurrentlyHere) {
			levSum += x.second;
		}
 
		for (int x: trieNow->indexesEndingHere) {
			massSearchResults[x] = levSum;
		}
 
		if (trieNow->sons.empty()) {
			return;
		}
 
		///transform trieNow->sons in a sorted array.
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


};
 

void parse_input(
    std::vector<std::string>& patterns,
    std::vector<std::string>& targets) {

	std::ios::sync_with_stdio(false); std::cin.tie(nullptr); std::cout.tie(nullptr);

    int q; std::cin >> q;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    
    std::string pat;
    while (q--) {
        std::getline(std::cin, pat);
        patterns.push_back(pat);
    }

    int count_s = 0;
    std::cin >> count_s;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::string s;
    for (int i = 0; i < count_s; ++i) {
        std::getline(std::cin, s);
        targets.push_back(s);
    }
}

int main() {

    std::vector<std::string> patterns, targets;
    parse_input(patterns, targets);

 	ExpoSizeStrSrc* E3S = new ExpoSizeStrSrc();

 	for (auto& p: patterns) {
 		E3S->insertQueriedString(p);
 	}

 	for (auto& t: targets) {
 		E3S->setupDag(t);
		E3S->massSearch(E3S->trieRoot);
		for (int x: E3S->massSearchResults) {
			std::cout << x << '\n';
		}
 	}
 
	delete E3S;
	return 0;
}

