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

struct FPll {ll fi, se;};

struct FPllPii {
  ll fi; FPii se;
  bool operator < (const FPllPii &oth) const { return fi < oth.fi || (fi == oth.fi && se < oth.se); }
};

///!!uses a..z. indexes from 1.
///first has to be int/ll. unstable sort.
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
  FPii cut(int l, int r) {
    return FPii(
      (hhPref[r].fi - (ll)hhPref[l-1].fi * p27[r-l+1].fi % mod.fi + mod.fi) % mod.fi,
      (hhPref[r].se - (ll)hhPref[l-1].se * p27[r-l+1].se % mod.se + mod.se) % mod.se
    );
  }
};

class ExpoSizeStrSrc {
private:
  const FPii mod = FPii(1000000007, 1000000009);
  const FPii arbBase = FPii(2097152, 4194304); ///8191, 524287
  static const int maxn = 100000, ml2 = 17;

  int g[maxn*ml2+1][ml2+1]; ///g[..][0] = size.
  int id[maxn+1][ml2+1]; ///id[i][j] = at what index in g will I find s[i..i+(1<<j)-1].
  FPii idToHhAsoc[maxn*ml2+1]; ///give DAG id => get associated hash for the respective node.
  int remainingIds[maxn*ml2+1]; ///what ids remain after compression? their number is kept in g[0][0].

  int leverage[maxn*ml2+1];
  FPii hhAsoc[maxn+1][ml2+1]; ///hhAsoc[i][j] = string.cut(i, i+(1<<j)-1).
  FPii hhG[maxn+1][ml2+1]; ///associated hash for the id[i][j] subtree.
  FPllPii hhGToId[maxn*ml2+1]; ///<hhG of (i, j), (i, j)>.
  bool gIdTakenHelp[maxn*ml2+1]; ///must look out for duplicates when eventually building g.

  ///trie built from dictionary entries.
  struct TrieNode {
    vi indexesEndingHere; ///the indexes whose dictionary strings end here.
    map<FPii, TrieNode *> sons; ///do I have a son with some associated hash?
    vector<FPii> idLevsCurrentlyHere; ///keep track of tokens that are in this trie node.
  };

  const int TNBufSz = 4096;
  int TNBufInd = TNBufSz;
  TrieNode *TNBuffer;

  TrieNode *trieNodeAlloc() {
    if (TNBufInd >= TNBufSz) {
      TNBuffer = new TrieNode[TNBufSz];
      TNBufInd = 0;
    }

    return &TNBuffer[TNBufInd++];
  }

public:
  ExpoSizeStrSrc(){}

  TrieNode *trieRoot = trieNodeAlloc();
  vi massSearchResults; ///results after mass-search. how many times does .. appear in s?

  ///builds the DAG for the given string.
  void init(string s) {
    int n = sz(s);
    HashedString hs(s, mod);

    trieRoot->idLevsCurrentlyHere.pb(FPii(0, inf));

    int i, j, z;
    int gNodeCnt = 1;
    for (i = 1; i <= n; i++) {
      for (j = 0; i+(1<<j)-1 <= n; j++) {
        leverage[gNodeCnt] = 1;
        hhAsoc[i][j] = hs.cut(i, i+(1<<j)-1);
        idToHhAsoc[gNodeCnt] = hhAsoc[i][j];
        id[i][j] = gNodeCnt++;
      }
    }

    ///build hashes for g's subtrees.
    FPll extraAdd;
    FPii pBase(1, 1);
    for (j = 0; (1<<j) <= n; j++) {
      for (i = 1; i+(1<<j)-1 <= n; i++) {
        extraAdd.fi = (ll)hhAsoc[i][j].fi * pBase.fi; ///pBase = arbBase ^ j.
        extraAdd.se = (ll)hhAsoc[i][j].se * pBase.se;

        z = j-1;
        while (z >= 0 && i+(1<<j)+(1<<z)-1 > n) z--;
        if (z >= 0) {
            extraAdd.fi += hhG[i+(1<<j)][z].fi;
            extraAdd.se += hhG[i+(1<<j)][z].se;
        }

        hhG[i][j].fi = extraAdd.fi % mod.fi;
        hhG[i][j].se = extraAdd.se % mod.se;
      }

      pBase.fi = (ll)pBase.fi * arbBase.fi % mod.fi;
      pBase.se = (ll)pBase.se * arbBase.se % mod.se;
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
          leverage[id[hhGToId[j].se.fi][hhGToId[j].se.se]] +=
            leverage[id[hhGToId[z].se.fi][hhGToId[z].se.se]];
          leverage[id[hhGToId[z].se.fi][hhGToId[z].se.se]] = 0;

          id[hhGToId[z].se.fi][hhGToId[z].se.se] = id[hhGToId[j].se.fi][hhGToId[j].se.se];
        }
      }
    }

    ///finally build the graph.
    int originalId = 1;
    for (i = 1; i <= n; i++) {
      for (j = 0; i+(1<<j)-1 <= n; j++, originalId++) {
        ///edges from g[0].
        if (leverage[originalId] != 0) {
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

  ///how many times does t appear in s?
  void insertQueriedString(string t) {
    massSearchResults.pb(0);
    if (sz(t) > maxn || sz(t) <= 0) return;

    HashedString ht(t, mod);

    int i, j, z;
    TrieNode *trieNow = trieRoot, *trieNext = NULL;
    for (i = ml2, z = 1; i >= 0; i--) {
      if (sz(t) & (1<<i)) {
        FPii hh = ht.cut(z, z+(1<<i)-1);
        z += (1<<i);

        auto it = trieNow->sons.find(hh);
        if (it != trieNow->sons.end()) {
          trieNow = it->se;
        } else {
          trieNext = trieNodeAlloc();
          trieNow->sons[hh] = trieNext;
          trieNow = trieNext;
        }
      }
    }

    trieNow->indexesEndingHere.pb(sz(massSearchResults) - 1);
  }

  ///propagates what is in the given trie node.
  void massSearch(TrieNode *trieNow) {
    if (!trieNow) return;

    int levSum = 0;
    for (FPii &x: trieNow->idLevsCurrentlyHere) {
      levSum += x.se;
    }

    for (int x: trieNow->indexesEndingHere) {
      massSearchResults[x] = levSum;
    }

    if (trieNow->sons.empty()) {
      return;
    }

    int i, nod, nn, levChain;

    ///transform trieNow->sons in a sorted array.
    vector<pair<FPii, TrieNode *>> sons;
    //sons.reserve(sz(trieNow->sons));
    for (auto &x: trieNow->sons) sons.pb(x);

    for (FPii &x: trieNow->idLevsCurrentlyHere) {
      nod = x.fi; levChain = x.se;

      for (i = 1; i <= g[nod][0]; i++) {
        if (nod == 0) nn = remainingIds[i]; else nn = g[nod][i];

        auto it = lower_bound(all(sons), make_pair(idToHhAsoc[nn], (TrieNode *)NULL),
                              [](const pair<FPii, TrieNode *> &a, const pair<FPii, TrieNode *> &b) {
          if (a.fi.fi != b.fi.fi) return a.fi.fi < b.fi.fi;
          return a.fi.se < b.fi.se;
        });

        if (it != sons.end() && it->fi == idToHhAsoc[nn]) {
          if (sz(it->se->idLevsCurrentlyHere) == 0 || it->se->idLevsCurrentlyHere.back().fi != nn) {
            it->se->idLevsCurrentlyHere.pb(FPii(nn, min(levChain, leverage[nn])));
          } else {
            ///duplicates may exist. possible to have multiple descendants with the same index after compression.
            ///because of the mode in which I compress (merge in the lowest index) => the indexes from it->se->idLevs.. are ordered increasingly.
            it->se->idLevsCurrentlyHere.back().se += min(levChain, leverage[nn]);
          }
        }
      }
    }

    for (auto &x: trieNow->sons) {
      if (!x.se->idLevsCurrentlyHere.empty()) {
        massSearch(x.se);
      }
    }
  }
};

ExpoSizeStrSrc E3S;

int main() {
  ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);

  string s; cin >> s;

  E3S.init(s);

  int q; cin >> q;

  string t;
  for (int _ = 0; _ < q; _++) {
    cin >> t;
    E3S.insertQueriedString(t);
  }

  E3S.massSearch(E3S.trieRoot);

  for (int x: E3S.massSearchResults) {
    cout << x << '\n';
  }

  return 0;
}
