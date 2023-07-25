///PTM
#pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math,O3")
#pragma GCC target("sse,sse2,sse3,sse3,sse4,popcnt,abm,mmx,avx,tune=native")
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <climits>
#include <cassert>
#include <chrono>
#include <vector>
#include <random>
#include <stack>
#include <cmath>
#include <map>
#include <set>
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

int main(int argc, char **argv) {
  auto start = std::chrono::steady_clock::now(), stop = std::chrono::steady_clock::now();

  ///argv[1] = n.
  ///argv[2] = m.
  ///argv[3] = tip distributie.
  ///argv[4] = mod alegere urmatoarea lungime (0: *2 + 1, 1: +1)

  int n = atoi(argv[1]);
  ll m = atoll(argv[2]);
  int tipD = atoi(argv[3]);
  int tipNext = atoi(argv[4]);

  vector<double> dist;
  if (tipD >= 1 && tipD <= 26) dist = vector<double>(tipD, 1.0 / tipD);
  if (tipD == 27) dist = {0.99, 0.01};
  if (tipD == 28) dist = {0.9, 0.1};
  if (tipD == 29) dist = {0.6, 0.2, 0.1, 0.05, 0.05};

  mt19937 mt = mt19937();
  mt.seed(0); //time(NULL).

  ///add 5 chars to s, t.
  string s, t, tmp;
  for (int _ = 0; _ < 5; _++) {
    double x = uniform_real_distribution<double>(0, 1)(mt);
    int i = 0;
    while (i < sz(dist)-1 && x > dist[i]) {
      x -= dist[i];
      i++;
    }

    s += ('a' + i);
  }

  for (int _ = 0; _ < 5; _++) {
    double x = uniform_real_distribution<double>(0, 1)(mt);
    int i = 0;
    while (i < sz(dist)-1 && x > dist[i]) {
      x -= dist[i];
      i++;
    }

    t += ('a' + i);
  }

  while (sz(t) < n) {
    tmp = s + t;
    s = t;
    t = tmp;
  }

  s = t.substr(0, n);
  HashedString hs(s, FPii(1000000007, 1000000009));

  set<pii> ss;

  int i, j, z;

  set<int> rem;
  for (i = 1; i <= n; i++) {
    rem.insert(i);
  }

  auto getNextLength = [&tipNext] (int z) {
    if (tipNext == 0) return z*2+1;
    return z + 1;
  };

  vector<pii> qrys;
  while (1) {
    if (m <= 0 || rem.empty()) break;
    z = *rem.begin();
    for (; z <= n; z = getNextLength(z)) {
      auto it = rem.find(z);
      if (it != rem.end()) rem.erase(it);
      for (i = 1; i <= n; i++) {
        j = i+z-1;
        if (j > n) break;
        FPii now;
        hs.cut(i, j, now);
        if (!ss.count(pii(now.fi, now.se))) {
          if (m < z) goto gata;
          m -= z;
          ss.insert(pii(now.fi, now.se));
          qrys.pb(pii(i, j));
        }
      }
    }
  }

  gata:

  cout << s << '\n' << sz(qrys) << '\n';
  for (pii now: qrys) {
    //cout << s.substr(now.fi-1, now.se-now.fi+1) << '\n';
    
    cout << now.fi << ' ' << now.se << '\n';
  }

  return 0;
}
