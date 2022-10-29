///https://cses.fi/problemset/hack/2103/entry/2893480/
#include <bits/stdc++.h>
using namespace std;

const int N = 1e5 + 5;

// Node 1 is the initial node of the automaton
const int SA = 2 * N;
int last = 1;
int len[SA], link[SA];
array<int, 26> to[SA];
int lastID = 1;

int cnt[SA];

void add(int c) {
    int u = ++lastID;
    len[u] = len[last] + 1;

    cnt[u]++;

    int p = last;
    last = u; // update last immediately
    for (; p > 0 && !to[p][c]; p = link[p])
        to[p][c] = u;

    if (p == 0) {
        // this was the first time the character c was added to the string
        link[u] = 1;
        return;
    }

    // there's a transition from a substring to { substring c } (from p to q)
    int q = to[p][c];
    if (len[q] == len[p] + 1) {
        // continuous transition
        // we can just add u to the endpos-set of q and be fine
        link[u] = q;
        return;
    }

    // non-continuous transition
    // we need to split q into two distinct endpos-sets
    int clone = ++lastID;
    len[clone] = len[p] + 1;
    link[clone] = link[q];
    link[q] = link[u] = clone;
    to[clone] = to[q];
    for (int pp = p; to[pp][c] == q; pp = link[pp])
        to[pp][c] = clone;
}

int main() {
	cin.tie(0)->sync_with_stdio(false);

	string s; cin >> s;
	for (char c : s)
		add(c - 'a');

	vector<int> nodes(lastID);

	iota(nodes.begin(), nodes.end(), 1);
	sort(nodes.begin(), nodes.end(),
			[&](int u, int v) { return len[u] > len[v]; });

	for (int u : nodes)
		cnt[link[u]] += cnt[u];

	cnt[0] = 0;

	int q; cin >> q;
	while (q--) {
		string t; cin >> t;

		int u = 1;
		for (char _c : t) {
			int c = _c - 'a';
			u = to[u][c];
		}
		cout << cnt[u] << '\n';
	}

	return 0;
}
