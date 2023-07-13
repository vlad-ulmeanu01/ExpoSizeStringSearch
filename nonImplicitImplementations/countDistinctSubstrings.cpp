///PTM HAI
#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math,O3")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")

#include <algorithm>
#include <iostream>
#include <fstream>
#include <climits>
#include <cassert>
#include <chrono>
#include <vector>
#include <stack>
#include <cmath>
#include <map>
#define ll long long
#define pii pair<int,int>
#define pli pair<ll,int>
#define pil pair<int,ll>
#define pll pair<ll,ll>
#define fi first
#define se second
#define inf (INT_MAX/2-1)
#define infl (1LL<<61)
#define vi vector<int>
#define vl vector<ll>
#define pb push_back
#define sz(a) ((int)(a).size())
#define all(a) begin(a),end(a)
#define y0 y5656
#define y1 y7878
#define aaa system("pause");
#define dbg(x) std::cerr<<(#x)<<": "<<(x)<<'\n',aaa
#define dbga(x,n) std::cerr<<(#x)<<"[]: ";for(int _=0;_<n;_++)std::cerr<<x[_]<<' ';std::cerr<<'\n',aaa
#define dbgs(x) std::cerr<<(#x)<<"[stl]: ";for(auto _:x)std::cerr<<_<<' ';std::cerr<<'\n',aaa
#define dbgp(x) std::cerr<<(#x)<<": "<<x.fi<<' '<<x.se<<'\n',aaa
#define dbgsp(x) std::cerr<<(#x)<<"[stl pair]:\n";for(auto _:x)std::cerr<<_.fi<<' '<<_.se<<'\n';aaa
#define TIMER(x) stop = std::chrono::steady_clock::now(); cerr << x << ' ' << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000000.0 << '\n'; start = std::chrono::steady_clock::now();

using namespace std;

struct FPii {
    int fi, se;
    FPii(int fi_, int se_){fi = fi_; se = se_;}
    FPii(){fi = se = 0;}
    bool operator == (const FPii &oth) const { return fi == oth.fi && se == oth.se; }
    bool operator < (const FPii &oth) const { return fi < oth.fi || (fi == oth.fi && se < oth.se); }
};

struct FPllPii {
    ll fi; FPii se;
    bool operator < (const FPllPii &oth) const { return fi < oth.fi || (fi == oth.fi && se < oth.se); }
};

///!!uses a..z. indexes from 1.

///first has to be int/ll. doesn't sort the second element in the pairs.
template<typename T>
void radixSortPairs (int l, int r, T *v) {
    const int base = 256;

    vector<vector<T>> u(2);
    u[0].resize(r+1); u[1].resize(r+1);
    int cnt[base] = {0};

    int i, j, z, pin;

    auto mel = min_element(v+l, v+r+1)->fi;
    if (mel > 0) mel = 0;

    for (i = l; i <= r; i++) {
        u[0][i].fi = v[i].fi - mel;
        u[0][i].se = v[i].se;
    }

    int noPasses = sizeof(v[l].fi); ///4 for int, 8 for ll.
    for (i = 0, pin = 0; i < noPasses; i++, pin ^= 1) {
        fill(cnt, cnt + base, 0);

        for (j = l; j <= r; j++) {
            cnt[(u[pin][j].fi >> (i << 3)) & 255]++;
        }

        for (j = 1; j < base; j++) {
            cnt[j] += cnt[j-1];
        }

        for (j = r; j >= l; j--) {
            z = ((u[pin][j].fi >> (i << 3)) & 255);
            u[pin^1][l + (--cnt[z])] = u[pin][j];
        }
    }

    for (i = l; i <= r; i++) {
        v[i].fi = u[pin][i].fi + mel;
        v[i].se = u[pin][i].se;
    }
}

template<typename T>
void radixSort(int l, int r, T *v) {
    const int base = 256;

    vector<vector<T>> u(2);
    u[0].resize(r+1); u[1].resize(r+1);
    int cnt[base] = {0};

    int i, j, z, pin;

    auto mel = *min_element(v+l, v+r+1);
    if (mel > 0) mel = 0;

    for (i = l; i <= r; i++) {
        u[0][i] = v[i] - mel;
    }

    int noPasses = sizeof(v[l]); ///4 for int, 8 for ll.
    for (i = 0, pin = 0; i < noPasses; i++, pin ^= 1) {
        fill(cnt, cnt + base, 0);

        for (j = l; j <= r; j++) {
            cnt[(u[pin][j] >> (i << 3)) & 255]++;
        }

        for (j = 1; j < base; j++) {
            cnt[j] += cnt[j-1];
        }

        for (j = r; j >= l; j--) {
            z = ((u[pin][j] >> (i << 3)) & 255);
            u[pin^1][l + (--cnt[z])] = u[pin][j];
        }
    }

    for (i = l; i <= r; i++) {
        v[i] = u[pin][i] + mel;
    }
}

struct HashedString {
    int n;
    string s;
    vector<FPii> p27, hhPref;
    FPii mod;

    HashedString(string s_, FPii mod_) {
        n = sz(s_);
        p27.resize(n+1);
        hhPref.resize(n+1);
        s = " " + s_;
        mod = mod_;

        p27[0] = FPii(1, 1);
        hhPref[0] = FPii(0, 0);
        for (int i = 1; i <= n; i++) {
            p27[i].fi = (ll)p27[i-1].fi * 27 % mod.fi;
            p27[i].se = (ll)p27[i-1].se * 27 % mod.se;

            hhPref[i].fi = ((ll)hhPref[i-1].fi * 27 + s[i]-'a'+1) % mod.fi;
            hhPref[i].se = ((ll)hhPref[i-1].se * 27 + s[i]-'a'+1) % mod.se;
        }
    }

    ///returns the corresponding hash for the [l..r] string.
    void cut(int l, int r, FPii &ans) {
        ans.fi = hhPref[r].fi - (ll)hhPref[l-1].fi * p27[r-l+1].fi % mod.fi + mod.fi;
        ans.se = hhPref[r].se - (ll)hhPref[l-1].se * p27[r-l+1].se % mod.se + mod.se;
        if (ans.fi >= mod.fi) ans.fi -= mod.fi;
        if (ans.se >= mod.se) ans.se -= mod.se;
    }
};

class ExpoSizeStrSrc {
private:
    const FPii mod = FPii(1'000'000'007, 1'000'000'009);
    static const int maxn = 100'000, ml2 = 17;

    int n; ///given string size;
    int g[maxn*ml2+1][ml2+1]; ///g[..][0] = size.
    int id[maxn+1][ml2+1]; ///id[i][j] = at what index in g will I find s[i..i+(1<<j)-1].
    ll idToHhAsoc[maxn*ml2+1]; ///give DAG id => get associated hash for the respective node.
    int remainingIds[maxn*ml2+1]; ///what ids remain after compression? their number is kept in g[0][0].

    int idToLength[maxn*ml2+1]; ///id => what is the length of the assoc. substring? ex ..[id[i][j]] = (1<<j).
    int idToIndexEnd[maxn*ml2+1]; ///id => where does the assoc. substring end in s? ex ..[id[i][j]] = i+(1<<j)-1.

    bool disappeared[maxn*ml2+1]; ///if a node id has been compressed into another, mark it as disappeared.
    FPii hhAsoc[maxn+1][ml2+1]; ///hhAsoc[i][j] = string.cut(i, i+(1<<j)-1).
    FPii hhG[maxn+1][ml2+1]; ///associated hash for the id[i][j] subtree.
    FPllPii hhGToId[maxn*ml2+1]; ///<hhG of (i, j), (i, j)>.
    bool gIdTakenHelp[maxn*ml2+1]; ///must look out for duplicates when eventually building g.

    ///used by the massSearch function.
    ll sonsHashes[maxn*ml2+1];
public:
    ExpoSizeStrSrc(){}

    ll cntDistinctSubstrings; ///results after mass-search.

    ///builds the DAG for the given string.
    void init(string s) {
        n = sz(s);
        HashedString hs(s, mod);

        cntDistinctSubstrings = 0;

        int i, j, z;
        int gNodeCnt = 1;
        for (i = 1; i <= n; i++) {
            for (j = 0; i+(1<<j)-1 <= n; j++) {
                hs.cut(i, i+(1<<j)-1, hhAsoc[i][j]);
                idToHhAsoc[gNodeCnt] = ((ll)hhAsoc[i][j].fi << 32) | hhAsoc[i][j].se;
                idToLength[gNodeCnt] = (1<<j);
                idToIndexEnd[gNodeCnt] = i + (1<<j) - 1;
                id[i][j] = gNodeCnt++;
            }
        }

        ///build hashes for g's subtrees.
        for (j = 0; (1<<j) <= n; j++) {
            for (i = 1; i+(1<<j)-1 <= n; i++) {
                hs.cut(i, min(n, i+(1<<(j+1))-2), hhG[i][j]);
            }
        }

        ///if there are 2 nodes in the DAG which share the same hhG (=> same hhAsoc as well), unite them.
        ///same hhAsoc => (high probability) same substring retained.
        ///same hhG => (high probability) exactly the same subtree.

        int szh = 0;
        for (i = 1; i <= n; i++) {
            for (j = 0; i+(1<<j)-1 <= n; j++) {
                hhGToId[szh].fi = ((ll)hhG[i][j].fi << 32) | hhG[i][j].se;
                hhGToId[szh].se = FPii(i, j);
                szh++;
            }
        }

        radixSortPairs<FPllPii>(0, szh-1, hhGToId);

        z = 0;
        while (z < szh) {
            i = j = z;
            while (i < szh && hhGToId[i].fi == hhGToId[z].fi) {
                if (hhGToId[i].se < hhGToId[j].se) {
                    j = i;
                }

                i++;
            }

            ///[z, i) have the same hhG. unite them, but merge in the node with the min id (j) (ie the one that appears first
            ///and in case of equality is the shortest). (helps determining the position of the first match).
            for (; z < i; z++) {
                if (z != j) {
                    disappeared[id[hhGToId[z].se.fi][hhGToId[z].se.se]] = true;
                    id[hhGToId[z].se.fi][hhGToId[z].se.se] = id[hhGToId[j].se.fi][hhGToId[j].se.se];
                }
            }
        }

        ///finally build the graph.
        int originalId = 1;
        for (i = 1; i <= n; i++) {
            for (j = 0; i+(1<<j)-1 <= n; j++, originalId++) {
                ///edges from g[0].
                if (!disappeared[originalId]) {
                    remainingIds[++g[0][0]] = originalId;
                }

                ///where can I go from s[i..i+(1<<j)-1]?
                if (g[id[i][j]][0] == 0) {
                    for (z = 0; z < j; z++) { ///!! z < j.
                        if (i+(1<<j)+(1<<z)-1 <= n && !gIdTakenHelp[id[i+(1<<j)][z]]) {
                            g[id[i][j]][0]++;
                            g[id[i][j]][g[id[i][j]][0]] = id[i+(1<<j)][z];
                            gIdTakenHelp[id[i+(1<<j)][z]] = true;
                        }
                    }

                    for (z = 1; z <= g[id[i][j]][0]; z++) {
                        gIdTakenHelp[g[id[i][j]][z]] = false;
                    }
                }
            }
        }
    }

    ///propagates what is in the given trie node.
    void massSearch(vi idsCurrentlyHere) {
        if (idsCurrentlyHere.empty()) {
            return;
        }

        ///all of the current node tokens show the same value of a subtring, just in different areas of the string.
        if (idsCurrentlyHere != vi{0}) { ///the trie root is a special case.
            cntDistinctSubstrings++;

            ///if I only have one token showing the value of a substring, I have a distinct prefix that can't be found
            ///anywhere else in the string, so I can just add all of its possible descendants now and not propagate
            ///anymore.
            if (sz(idsCurrentlyHere) == 1) {
                int x = idsCurrentlyHere[0];
                cntDistinctSubstrings += min(idToLength[x] - 1, n - idToIndexEnd[x]);
                return;
            }
        }

        int i, j, nn;

        ///each child has another associated hash. ex I am in [aaaa]bcd and I have 2 children, HASH(bc) and HASH(b).
        ///keep for each child a list of ids (tokens) that it has and must be propagated further.

        int cntSons = 0;
        for (int nod: idsCurrentlyHere) {
            for (i = 1; i <= g[nod][0]; i++) {
                if (nod == 0) nn = remainingIds[i]; else nn = g[nod][i];
                sonsHashes[cntSons++] = idToHhAsoc[nn];
            }
        }

        radixSort<ll>(0, cntSons-1, sonsHashes);
        cntSons = unique(sonsHashes, sonsHashes + cntSons) - sonsHashes;

        vector<vi> sons(cntSons);
        for (int nod: idsCurrentlyHere) {
            for (i = 1; i <= g[nod][0]; i++) {
                if (nod == 0) nn = remainingIds[i]; else nn = g[nod][i];

                ///j = the index in which nn will end up.
                j = lower_bound(sonsHashes, sonsHashes + cntSons, idToHhAsoc[nn]) - sonsHashes;

                ///duplicates may exist. possible to have multiple descendants with the same index after compression.
                ///because of the mode in which I compress (merge in the lowest index) => the indexes arrive in
                ///increasing order.
                if (sons[j].empty() || sons[j].back() != nn) {
                    sons[j].pb(nn);
                }
            }
        }

        for (vi &x: sons) {
            if (!x.empty()) {
                massSearch(x);
            }
        }
    }
};

ExpoSizeStrSrc E3S;

int main() {
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);

    string s; cin >> s;

    E3S.init(s);
    E3S.massSearch(vi{0});

    cout << E3S.cntDistinctSubstrings << '\n';

    return 0;
}
