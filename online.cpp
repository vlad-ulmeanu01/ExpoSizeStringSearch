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
    u[0][i] = make_pair(v[i].fi - mel, v[i].se);
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
    v[i] = make_pair(u[pin][i].fi + mel, u[pin][i].se);
  }
}

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
  const pii arbBase = pii(8191, 524287);
  static const int maxn = 100000, ml2 = 17;

  pii pBase[ml2+1]; ///pBase[i] = (arbBase.fi^i, arbBase.se^i).
  int g[maxn*ml2+1][ml2+1]; ///g[..][0] = sz(g[..]).
  int id[maxn+1][ml2+1]; ///id[i][j] = at what index in g will I find s[i..i+(1<<j)-1].
  pii hhAsoc[maxn+1][ml2+1]; ///hhAsoc[i][j] = string.cut(i, i+(1<<j)-1).
  pii hhG[maxn+1][ml2+1]; ///associated hash for the id[i][j] subtree.
  pii idToHhAsoc[maxn*ml2+1]; ///id from graph => associated hash for that node.
  pli hhAsocToIds[maxn*ml2+1]; ///take associated hash => what ids from g have that hash.
  int szHhAsocToIds = 0; ///size for ^.
  bool gIdTakenHelp[maxn*ml2+1]; ///must look out for duplicates when eventually building g.
  int leverage[maxn*ml2+1]; ///when uniting two nodes, l[x] += l[y]+1, l[y] = 0. used for queryCnt.

  ///{hhG/hhGt pentru (i, j), (i, j) = poz in graf} (nu conteaza hhAsoc ptc e incorporat in hhG/hhGt).
  pair<ll, pii> hhGToId[maxn*ml2+1]; ///<hhG of (i, j), (i, j)>.

  int idToPos[maxn*ml2+1]; ///at what position in s does the substring begin? (for first position query)

  ///complimentary trie.
  ///fill up trie as queries arrive. now, resolving a query consists in walking through the trie to determine the longest common
  ///prefix (that the construction permits) with a previous query + continuing to build the trie.
  struct TrieNode {
    map<pii, TrieNode *> sons; ///(hhAsoc son, asociated address).

    ///have multiple chains that lead here.
    vector<pair<pii, int>> idLevSPOfChainsLeadingHere; ///(current node id, min(leverage[..] | .. is part of current chain)), start positions that end up here.
    ///current chain ends here.

    int leveragesSum; ///sum of .se for the previous array.

    TrieNode () {
      leveragesSum = 0;
    }

    ///returns the coresponding address for hhAsoc, if it exists in the trie node.
    TrieNode *findAdressOfHhAsoc(pii hh) {
      auto it = sons.find(hh);
      if (it == sons.end()) return NULL;
      return it->se;
    }

    ///inserts a hhAsoc. returns its address. if it already exists, it just returns the address.
    TrieNode *insertHhAsoc(pii hh) {
      TrieNode *now = findAdressOfHhAsoc(hh);
      if (!now) {
        now = new TrieNode;
        sons[hh] = now;
      }

      return now;
    }
  };

  TrieNode *trieRoot = new TrieNode;

  ///goes as far down in the trie as possible. pulls out hashes from the
  ///stack as the search continues.
  TrieNode *
  walkThruTrie(TrieNode *trieNow,
               stack<pii> &hhChain)
  {
    TrieNode *trieNext = NULL;

    while (!hhChain.empty()) {
      trieNext = trieNow->findAdressOfHhAsoc(hhChain.top());
      if (trieNext) {
        hhChain.pop();
        trieNow = trieNext;
      } else {
        break;
      }
    }

    return trieNow;
  }

  void queryNormalDf(TrieNode *trieNow, stack<pii> &hhChain, int leverageMin, int nod, int startingPos) {
    trieNow = trieNow->insertHhAsoc(hhChain.top());
    leverageMin = min(leverageMin, leverage[nod]);
    trieNow->leveragesSum += leverageMin;
    trieNow->idLevSPOfChainsLeadingHere.pb(make_pair(pii(nod, leverageMin), startingPos));

    hhChain.pop();

    if (hhChain.empty()) {
      return;
    }

    int i, nn;
    pii hhTop = hhChain.top();
    for (i = 1; i <= g[nod][0]; i++) {
      nn = g[nod][i];
      if (idToHhAsoc[nn] == hhTop) {
        queryNormalDf(trieNow, hhChain, leverageMin, nn, startingPos);
        hhChain.push(hhTop);
      }
    }
  }

  void queryPrefixedDf(TrieNode *trieNow, stack<pii> &hhChain, int leverageMin, int nod, int startingPos) {
    if (hhChain.empty()) {
      return;
    }

    pii hhTop = hhChain.top();
    TrieNode *trieNext = NULL;

    int i, nn;
    for (i = 1; i <= g[nod][0]; i++) {
      nn = g[nod][i];
      if (idToHhAsoc[nn] == hhTop) {
        if (!trieNext) {
          trieNext = trieNow->insertHhAsoc(hhTop);
        }

        trieNext->leveragesSum += min(leverageMin, leverage[nn]);
        trieNext->idLevSPOfChainsLeadingHere.pb(make_pair(pii(nn, min(leverageMin, leverage[nn])), startingPos));

        hhChain.pop();
        queryPrefixedDf(trieNext, hhChain, min(leverageMin, leverage[nn]), nn, startingPos);
        hhChain.push(hhTop);
      }
    }
  }

public:
  ExpoSizeStrSrc(){}

  ///builds the DAG for the given string.
  void init(string s) {
    int n = sz(s);
    HashedString hs(s, mod);

    int i, j, z;

    pBase[0] = pii(1, 1);
    for (i = 1; i <= ml2; i++) {
      pBase[i].fi = (ll)pBase[i-1].fi * arbBase.fi % mod.fi;
      pBase[i].se = (ll)pBase[i-1].se * arbBase.se % mod.se;
    }

    int gNodeCnt = 0;
    for (i = 1; i <= n; i++) {
      for (j = 0; i+(1<<j)-1 <= n; j++) {
        leverage[gNodeCnt] = 1;
        idToPos[gNodeCnt] = i;
        id[i][j] = gNodeCnt++;
        hhAsoc[i][j] = hs.cut(i, i+(1<<j)-1);
      }
    }

    ///build hashes for g's subtrees.
    for (j = 0; (1<<j) <= n; j++) {
      for (i = 1; i+(1<<j)-1 <= n; i++) {
        hhG[i][j].fi = (ll)hhAsoc[i][j].fi * pBase[j].fi % mod.fi;
        hhG[i][j].se = (ll)hhAsoc[i][j].se * pBase[j].se % mod.se;
        for (z = j-1; z >= 0; z--) {
          if (i+(1<<j)+(1<<z)-1 <= n) {
            hhG[i][j].fi = (hhG[i][j].fi + hhG[i+(1<<j)][z].fi) % mod.fi;
            hhG[i][j].se = (hhG[i][j].se + hhG[i+(1<<j)][z].se) % mod.se;
          }
        }
      }
    }

    ///if there are 2 nodes in the DAG which share the same hhG (=> same hhAsoc as well), unite them.
    ///same hhAsoc => (high probability) same substring retained.
    ///same hhG => (high probability) exactly the same subtree.

    int szh = 0;
    for (i = 1; i <= n; i++) {
      for (j = 0; i+(1<<j)-1 <= n; j++) {
        hhGToId[szh].fi = ((ll)hhG[i][j].fi << 32) | hhG[i][j].se;
        hhGToId[szh].se = pii(i, j);
        szh++;
      }
    }

    radixSortPairs<pair<ll, pii>>(0, szh-1, hhGToId);

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
    for (i = 1; i <= n; i++) {
      for (j = 0; i+(1<<j)-1 <= n; j++) {
        ///where can I go from s[i..i+(1<<j)-1]?

        if (g[id[i][j]][0] == 0) {
          for (z = 0; z < j; z++) { ///!! z < j.
            if (i+(1<<j)+(1<<z)-1 <= n) {
              if (!gIdTakenHelp[id[i+(1<<j)][z]]) {
                g[id[i][j]][0]++;
                g[id[i][j]][g[id[i][j]][0]] = id[i+(1<<j)][z];
                gIdTakenHelp[id[i+(1<<j)][z]] = true;
              }
            }
          }

          for (z = 1; z <= g[id[i][j]][0]; z++) {
            gIdTakenHelp[g[id[i][j]][z]] = false;
          }

          idToHhAsoc[id[i][j]] = hhAsoc[i][j];
          hhAsocToIds[szHhAsocToIds++] = pli((ll)hhAsoc[i][j].fi * mod.se + hhAsoc[i][j].se, id[i][j]);
        }
      }
    }

    radixSortPairs<pli>(0, szHhAsocToIds-1, hhAsocToIds);
    szHhAsocToIds = unique(hhAsocToIds, hhAsocToIds+szHhAsocToIds) - hhAsocToIds;
  }

  ///fi = number of occurences of t as a substring in s.
  ///se = position of first occurence of t in s.
  pii query(string t) {
    if (sz(t) > maxn || sz(t) <= 0) return pii(0, -1);

    HashedString ht(t, mod);

    int nt = sz(t);
    stack<pii> hhChain;

    int i = 0;
    while (nt > 0) {
      if (nt & (1<<i)) {
        hhChain.push(ht.cut(nt-(1<<i)+1, nt));
        nt ^= (1<<i);
      }
      i++;
    }

    ///walk through the trie to see have common prefix I have.
    TrieNode *trieNow = walkThruTrie(trieRoot, hhChain);

    ///if the common string <=> whole wanted string => already searched for it.
    if (hhChain.empty()) {
      return pii(
        trieNow->leveragesSum,
        trieNow->idLevSPOfChainsLeadingHere[0].se
      );
    }

    if (trieNow == trieRoot) {
      ///trieNow == trieRoot, do normal search, have nothing in common in trie with my query.

      ///need to find a chain which contains exactly the hashes from hhChain.
      pii hhTop = hhChain.top();

      ///search for the interval from hhAsocToIds. what ids correspond to the nodes from the graph with the wanted hhTop hash?
      ll hhAsocLL = (ll)hhTop.fi * mod.se + hhTop.se;

      for (i = lower_bound(hhAsocToIds, hhAsocToIds+szHhAsocToIds, pli(hhAsocLL, -inf)) - hhAsocToIds;
           i < szHhAsocToIds && hhAsocToIds[i].fi == hhAsocLL;
           i++) {
        int nod = hhAsocToIds[i].se;

        queryNormalDf(trieNow, hhChain, inf, nod, idToPos[nod]);
        hhChain.push(hhTop);
      }
    } else {
      ///won't search in hhAsocToIds, use trieNow->idLevSPOfChainsLeadingHere.
      for (auto &x: trieNow->idLevSPOfChainsLeadingHere) {
        ///x.fi.fi = node at the end of the chain,
        ///x.fi.se = minimum leverage value across the current chain,
        ///x.se = index from where the chain starts.
        queryPrefixedDf(trieNow, hhChain, x.fi.se, x.fi.fi, x.se);
      }
    }

    trieNow = walkThruTrie(trieNow, hhChain);
    if (hhChain.empty()) {
      return pii(
        trieNow->leveragesSum,
        trieNow->idLevSPOfChainsLeadingHere[0].se
      );
    }

    ///if hhChain is still not empty, we couldn't find the wanted string.
    return pii(0, -1);
  }
};

ExpoSizeStrSrc E3S;

int main() {
  ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);

  string s; cin >> s;
  E3S.init(s);

  int q; cin >> q;
  string t;
  while (q--) {
    cin >> t;
    pii sol = E3S.query(t); ///apparition count, position of first apparition.
    cout << sol.fi << '\n';
  }

  return 0;
}
