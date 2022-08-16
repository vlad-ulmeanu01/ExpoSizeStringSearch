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

///!!folosire ch a..z. indexare de la 1.
///first tb sa fie int/ll. ordinea lui second este incerta dupa sortare.
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

  int noPasses = sizeof(v[l].fi); ///pt int 4, pt ll 8.
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

  ///returneaza hashul corespunzator stringului [l..r].
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


  int g[maxn*ml2+1][ml2+1]; ///g[..][0] = size.
  int id[maxn+1][ml2+1]; ///id[i][j] = la ce indice in g o sa gasesc s[i..i+(1<<j)-1].
  pii idToHhAsoc[maxn*ml2+1]; ///dau id din graf => hh asociat nodului respectiv.
  int remainingIds[maxn*ml2+1]; ///dupa compresie, ce id-uri raman? numarul lor e stocat in g[0][0].

  pii pBase[ml2+1];
  int leverage[maxn*ml2+1];
  pii hhAsoc[maxn+1][ml2+1]; ///hhAsoc[i][j] = string.cut(i, i+(1<<j)-1).
  pii hhG[maxn+1][ml2+1]; ///hhG[i][j] = hashul asociat subarborelui id[i][j].
  pair<ll, pii> hhGToId[maxn*ml2+1]; ///hhG al lui (i, j) => ce (i, j) are hashul ala.
  bool gIdTakenHelp[maxn*ml2+1]; ///cand construiesc g, trb sa am grija la duplicate.

  ///fac trie din stringurile pe care vreau sa le caut.
  struct TrieNode {
    vi indexesEndingHere; ///indicii ale caror stringuri cautate se termina aici.
    map<pii, TrieNode *> sons; ///hash asociat => am fiu?
    vector<pii> idLevsCurrentlyHere; ///in timpul cautarii. tine minte indicii care pot fi impinsi in jos, min lev pe lantul resp.
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
  vi massSearchResults; ///rezultatele dupa cautarea in masa. (de cate ori apare .. in s?).

  ///creeaza DAG-ul pentru un string.
  void init(string s) {
    int n = sz(s);
    HashedString hs(s, mod);

    trieRoot->idLevsCurrentlyHere.pb(pii(0, inf));

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

    ///copnstruiesc hashuri pentru subarborii lui g.
    pBase[0] = pii(1, 1);
    for (i = 1; i <= ml2; i++) {
      pBase[i].fi = (ll)pBase[i-1].fi * arbBase.fi % mod.fi;
      pBase[i].se = (ll)pBase[i-1].se * arbBase.se % mod.se;
    }

    pll extraAdd;
    for (j = 0; (1<<j) <= n; j++) {
      for (i = 1; i+(1<<j)-1 <= n; i++) {
        hhG[i][j].fi = (ll)hhAsoc[i][j].fi * pBase[j].fi % mod.fi;
        hhG[i][j].se = (ll)hhAsoc[i][j].se * pBase[j].se % mod.se;
        for (extraAdd.fi = extraAdd.se = 0, z = j-1; z >= 0; z--) {
          if (i+(1<<j)+(1<<z)-1 <= n) {
            extraAdd.fi += hhG[i+(1<<j)][z].fi;
            extraAdd.se += hhG[i+(1<<j)][z].se;
          }
        }
        extraAdd.fi %= mod.fi; extraAdd.se %= mod.se;
        hhG[i][j].fi += extraAdd.fi; if (hhG[i][j].fi >= mod.fi) hhG[i][j].fi -= mod.fi;
        hhG[i][j].se += extraAdd.se; if (hhG[i][j].se >= mod.se) hhG[i][j].se -= mod.se;
      }
    }

    ///daca exista 2 noduri in DAG care au acelasi hhAsoc si au acelasi hhG, le unesc.
    ///acelasi hhAsoc => (prob mare) acelasi substring retinut (si important aceeasi lg a substringului).
    ///acelasi hhG => (prob mare) exact acelasi subarbore.

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

      ///[z, i) au acelasi hhG. le unesc, insa pastrez nodul cu id-ul minim (ie cel care apare primul
      ///si in caz de egalitate este cel mai scurt. (ajuta pt determinarea primei aparitii).
      for (; z < i; z++) {
        if (z != j) {
          leverage[id[hhGToId[j].se.fi][hhGToId[j].se.se]] +=
            leverage[id[hhGToId[z].se.fi][hhGToId[z].se.se]];
          leverage[id[hhGToId[z].se.fi][hhGToId[z].se.se]] = 0;

          id[hhGToId[z].se.fi][hhGToId[z].se.se] = id[hhGToId[j].se.fi][hhGToId[j].se.se];
        }
      }
    }

    ///acum construiesc graful.
    int originalId = 1;
    for (i = 1; i <= n; i++) {
      for (j = 0; i+(1<<j)-1 <= n; j++, originalId++) {
        ///ma ocup de conexiunile din g[0].
        if (leverage[originalId] != 0) {
          remainingIds[++g[0][0]] = originalId;
        }

        ///in cine pot sa merg din s[i..i+(1<<j)-1]?
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

  ///de cate ori apare t in s?
  void insertQueriedString(int l, int r, HashedString &ht) {
    massSearchResults.pb(0);
    int nt = r-l+1;
    if (nt > maxn || nt <= 0) return;

    int i, j, z;
    TrieNode *trieNow = trieRoot, *trieNext = NULL;
    for (i = ml2, z = l; i >= 0; i--) {
      if (nt & (1<<i)) {
        pii hh = ht.cut(z, z+(1<<i)-1);
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

  ///primeste indicele nodului din g. propaga ce e pe acolo.
  int cnt = 0;

  ///primeste indicele nodului din g. propaga ce e pe acolo.
  void massSearch(TrieNode *trieNow) {
    if (!trieNow) return;

    int levSum = 0;
    for (pii &x: trieNow->idLevsCurrentlyHere) {
      levSum += x.se;
    }

    for (int x: trieNow->indexesEndingHere) {
      massSearchResults[x] = levSum;
    }

    if (trieNow->sons.empty()) {
      return;
    }

    int i, nod, nn, levChain;

    ///transform trieNow->sons intr-un vector sortat.
    vector<pair<pii, TrieNode *>> sons; for (auto &x: trieNow->sons) sons.pb(x);

    for (pii &x: trieNow->idLevsCurrentlyHere) {
      nod = x.fi; levChain = x.se;

      for (i = 1; i <= g[nod][0]; i++) {
        cnt++;

        if (nod == 0) nn = remainingIds[i]; else nn = g[nod][i];

        auto it = lower_bound(all(sons), make_pair(idToHhAsoc[nn], (TrieNode *)NULL),
                              [](const pair<pii, TrieNode *> &a, const pair<pii, TrieNode *> &b) {
          if (a.fi.fi != b.fi.fi) return a.fi.fi < b.fi.fi;
          return a.fi.se < b.fi.se;
        });

        if (it != sons.end() && it->fi == idToHhAsoc[nn]) {
          if (sz(it->se->idLevsCurrentlyHere) == 0 || it->se->idLevsCurrentlyHere.back().fi != nn) {
            it->se->idLevsCurrentlyHere.pb(pii(nn, min(levChain, leverage[nn])));
          } else {
            ///posibil sa am duplicate. posibil sa am mai multi descendenti cu acelasi indice dupa compresie.
            ///datorita modului in care fac compresia (unire in cel mai din stanga indice) => indicii din it->se->idLevs.. sunt crescatori.
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

void test100ka() {
  auto start = std::chrono::steady_clock::now(), stop = std::chrono::steady_clock::now();

  string s;
  int n = 100000;

  for (int _ = 0; _ < n; _++) {
    s += 'a';
  }

  E3S.init(s);

  TIMER('A')

  HashedString hs(s, pii(1000000007, 1000000009));

  for (int _ = 1; _ <= n; _++) {
    E3S.insertQueriedString(1, _, hs);
  }

  TIMER('B')

  E3S.massSearch(E3S.trieRoot);

  TIMER('C')
  cerr << E3S.cnt << '\n';
}

void test50kab49999a() {
  auto start = std::chrono::steady_clock::now(), stop = std::chrono::steady_clock::now();

  string s;
  int n = 100000;

  for (int _ = 0; _ < (n>>1); _++) s += 'a';
  s += 'b';
  for (int _ = 0; _ < (n>>1)-1; _++) s += 'a';

  E3S.init(s);

  TIMER('A')

  HashedString hs(s, pii(1000000007, 1000000009));

  for (int _ = 1; _ <= (n>>1); _++) {
    E3S.insertQueriedString(1, _, hs);
  }

  for (int l = (n>>1)+1; l >= 1; l--) {
    for (int r = (n>>1)+1; r <= (n>>1)+30; r++) {
      E3S.insertQueriedString(l, r, hs);
    }
  }

  TIMER('B')

  E3S.massSearch(E3S.trieRoot);

  TIMER('C')
  cerr << E3S.cnt << '\n';
}

void test50kab() {
  auto start = std::chrono::steady_clock::now(), stop = std::chrono::steady_clock::now();

  string s;
  int n = 100000;

  for (int _ = 0; _ < (n>>2); _++) {
    s += "abcd";
  }

  E3S.init(s);

  TIMER('A')

  HashedString hs(s, pii(1000000007, 1000000009));

  for (int i = 1; i <= 4; i++) {
    for (int _ = i; _ <= n; _++) {
      E3S.insertQueriedString(i, _, hs);
    }
  }

  TIMER('B')

  E3S.massSearch(E3S.trieRoot);

  TIMER('C')
  cerr << E3S.cnt << '\n';
}

void testAlmostPeriod() {
  auto start = std::chrono::steady_clock::now(), stop = std::chrono::steady_clock::now();

  string s;
  int n = 100000; ///de fapt o sa am n-1 ch in s.

  for (int _ = 0; _ < (n>>1) - 1; _++) {
    s += "ab";
  }

  s += "c";
  E3S.init(s);

  TIMER('A')

  HashedString hs(s, pii(1000000007, 1000000009));

  for (int i = 1; i <= 2; i++) {
    for (int _ = i; _ <= n-2; _++) {
      E3S.insertQueriedString(i, _, hs);
    }
  }

  for (int i = 1; i <= n-1; i++) {
    E3S.insertQueriedString(i, n-1, hs);
  }

  TIMER('B')

  E3S.massSearch(E3S.trieRoot);

  TIMER('C')
  cerr << E3S.cnt << '\n';
}

int main() {
  ///in tests 1, 3, 4 dictionaries are fully filled with all substrings that could be found in s.
//  test100ka();       ///aa..a (n = 1e5)
//  test50kab49999a(); ///one 'b' in middle of (1e5-1)'a's. dictionary is not completely filled.
//  test50kab();      ///abcdabcd...abcd (n = 1e5)
  testAlmostPeriod(); ///ababab...abc (n = 1e5-1).

  return 0;
}
