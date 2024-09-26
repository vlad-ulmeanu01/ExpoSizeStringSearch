///https://cses.fi/problemset/hack/2103/entry/2893480/
#include <bits/stdc++.h>
using namespace std;
 
const int N = 1e5 + 5;
 
// Node 1 is the initial node of the automaton
const int SA = 2 * N;
int last = 1;
int len[SA], Link[SA];
array<int, 26> to[SA];
int lastID = 1;
 
int cnt[SA];
 
void add(int c) {
    int u = ++lastID;
    len[u] = len[last] + 1;
 
	cnt[u]++;
 
    int p = last;
    last = u; // update last immediately
    for (; p > 0 && !to[p][c]; p = Link[p])
        to[p][c] = u;
 
    if (p == 0) {
        // this was the first time the character c was added to the string
        Link[u] = 1;
        return;
    }
 
    // there's a transition from a substring to { substring c } (from p to q)
    int q = to[p][c];
    if (len[q] == len[p] + 1) {
        // continuous transition
        // we can just add u to the endpos-set of q and be fine
        Link[u] = q;
        return;
    }
 
    // non-continuous transition
    // we need to split q into two distinct endpos-sets
    int clone = ++lastID;
    len[clone] = len[p] + 1;
    Link[clone] = Link[q];
    Link[q] = Link[u] = clone;
    to[clone] = to[q];
    for (int pp = p; to[pp][c] == q; pp = Link[pp])
        to[pp][c] = clone;
}

void reset() {
	last = 1; lastID = 1;

    // Reset len and Link arrays to 0
    std::fill(len, len + SA, 0);
    std::fill(Link, Link + SA, 0);
    std::fill(cnt, cnt + SA, 0);

    // Reset to array to -1 (or 0 if needed)
    for (int i = 0; i < SA; ++i) {
        to[i].fill(0); // Or .fill(0) if you need to reset to 0
    }
}

void parse_input(
    std::vector<std::string>& patterns,
    std::vector<std::string>& targets) {

    std::ios::sync_with_stdio(false); std::cin.tie(nullptr); std::cout.tie(nullptr);

    int q; cin >> q;
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
 
	// string s; //cin >> s;
    // std::getline(std::cin, s);

    for (auto& t: targets) {
    	reset();

		for (char c : t)
			add(c - 'a');

		vector<int> nodes(lastID);

		iota(nodes.begin(), nodes.end(), 1);
		sort(nodes.begin(), nodes.end(), [&](int u, int v) { return len[u] > len[v]; });

		for (int u : nodes)
			cnt[Link[u]] += cnt[u];
 	
		cnt[0] = 0;
 		
 		int i = 0; 
 		for (auto& p: patterns) {
 			int u = 1;
 			// std::cerr << i++ << "\n";

			for (char _c : p) {
				int c = _c - 'a';
				u = to[u][c];
			}

			cout << cnt[u] << '\n';
 		}

    }

	// for (char c : s)
	// 	add(c - 'a');
 
	// vector<int> nodes(lastID);

	// iota(nodes.begin(), nodes.end(), 1);
	// sort(nodes.begin(), nodes.end(),
	// 		[&](int u, int v) { return len[u] > len[v]; });

	// for (int u : nodes)
	// 	cnt[Link[u]] += cnt[u];
 
	// cnt[0] = 0;
 
	// int q; cin >> q;
	// while (q--) {
	// 	string t;
	// 	cin >> t;
	// 	// int l, r; cin >> l >> r;
	// 	// t = s.substr(l-1, r-l+1);

	// 	int u = 1;
	// 	for (char _c : t) {
	// 		int c = _c - 'a';
	// 		u = to[u][c];
	// 	}
	// 	cout << cnt[u] << '\n';
	// }
 
	return 0;
}
