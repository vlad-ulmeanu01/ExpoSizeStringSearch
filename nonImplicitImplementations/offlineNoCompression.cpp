///PTM
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

///!!uses a..z. indexes from 1.
struct HashedString {
  int n;
  string s;
  vector<pii> p27, hhPref;
  pii mod;

  HashedString(string s_, pii mod_) {
    n = sz(s_);
    p27.resize(n+1);
    hhPref.resize(n+1);
    s = " " + s_;
    mod = mod_;

    p27[0] = pii(1, 1);
    hhPref[0] = pii(0, 0);
    for (int i = 1; i <= n; i++) {
      p27[i].fi = (ll)p27[i-1].fi * 27 % mod.fi;
      p27[i].se = (ll)p27[i-1].se * 27 % mod.se;

      hhPref[i].fi = ((ll)hhPref[i-1].fi * 27 + s[i]-'a'+1) % mod.fi;
      hhPref[i].se = ((ll)hhPref[i-1].se * 27 + s[i]-'a'+1) % mod.se;
    }
  }

  ///returns the corresponding hash for the [l..r] string.
  pii cut(int l, int r) {
    return pii(
      (hhPref[r].fi - (ll)hhPref[l-1].fi * p27[r-l+1].fi % mod.fi + mod.fi) % mod.fi,
      (hhPref[r].se - (ll)hhPref[l-1].se * p27[r-l+1].se % mod.se + mod.se) % mod.se
    );
  }
};

class ExpoSizeStrSrc {
private:
  const pii mod = pii(1000000007, 1000000009);
  static const int maxn = 1000000, ml2 = 20;

  vi g[maxn*ml2+1];
  int id[maxn+1][ml2+1]; ///id[i][j] = at what index in g will I find s[i..i+(1<<j)-1].
  pii idToHhAsoc[maxn*ml2+1]; ///give DAG id => get associated hash for the respective node.

  ///trie built from dictionary entries.
  struct TrieNode {
    vi indexesEndingHere; ///the indexes whose dictionary strings end here.
    map<pii, TrieNode *> sons; ///do I have a son with some associated hash?
    vi idsCurrentlyHere; ///keep track of tokens that are in this trie node.
  };

public:
  ExpoSizeStrSrc(){}

  TrieNode *trieRoot = new TrieNode;
  vi massSearchResults; ///results after mass-search. how many times does .. appear in s?

  ///builds the DAG for the given string.
  void init(string s) {
    int n = sz(s);
    HashedString hs(s, mod);

    g[0].reserve(maxn*ml2+1);
    trieRoot->idsCurrentlyHere.pb(0);

    int i, j, z;
    int gNodeCnt = 1;
    for (i = 1; i <= n; i++) {
      for (j = 0; i+(1<<j)-1 <= n; j++) {
        g[0].pb(gNodeCnt);
        g[gNodeCnt].reserve(ml2+1);
        idToHhAsoc[gNodeCnt] = hs.cut(i, i+(1<<j)-1);
        id[i][j] = gNodeCnt++;
      }
    }

    for (i = 1; i <= n; i++) {
      for (j = 0; i+(1<<j)-1 <= n; j++) {
        ///where can I go from s[i..i+(1<<j)-1]?
        for (z = 0; z < j; z++) { ///!! z < j.
          if (i+(1<<j)+(1<<z)-1 <= n) {
            g[id[i][j]].pb(id[i+(1<<j)][z]);
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
        pii hh = ht.cut(z, z+(1<<i)-1);
        z += (1<<i);

        auto it = trieNow->sons.find(hh);
        if (it != trieNow->sons.end()) {
          trieNow = it->se;
        } else {
          trieNext = new TrieNode;
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

    for (int x: trieNow->indexesEndingHere) {
      massSearchResults[x] = sz(trieNow->idsCurrentlyHere);
    }

    for (int nod: trieNow->idsCurrentlyHere) {
      for (int nn: g[nod]) {
        auto it = trieNow->sons.find(idToHhAsoc[nn]);
        if (it != trieNow->sons.end()) {
          it->se->idsCurrentlyHere.pb(nn);
        }
      }
    }

    for (auto &x: trieNow->sons) {
      if (!x.se->idsCurrentlyHere.empty()) {
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
