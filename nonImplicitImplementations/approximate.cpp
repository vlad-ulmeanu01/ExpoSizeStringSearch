#include <bits/stdc++.h>
#define ll long long
#define pii pair<int,int>
#define pll pair<ll,ll>
#define pli pair<ll,int>
#define pil pair<int,ll>
#define fi first
#define se second
#define inf (INT_MAX/2-1)
#define infl (1LL<<60)
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
#define maxn 100000

using namespace std;

const pii mod = pii(1000000007, 1000000009);

pii p27[maxn+5];
vi g[18*maxn+5];

struct hasher {
  int n;
  string s;
  vector<pii> pref;

  hasher (string s_, int n_) {
    s = s_;
    n = n_;
    assert(sz(s) == n+1); /// " " + ...

    pref.resize(n+1, pii(0, 0));
    for (int i = 1; i <= n; i++) {
      pref[i].fi = ((ll)pref[i-1].fi * 27 + s[i] - 'a' + 1) % mod.fi;
      pref[i].se = ((ll)pref[i-1].se * 27 + s[i] - 'a' + 1) % mod.se;
      ///daca compari hashuri de lg diferita NU ai voie sa lasi 'a' <=> 0!!!!!!!!!!!
    }
  }

  pii cut (int l, int r) {
    if (l > r) swap(l, r);

    assert(1 <= l);
    assert(r <= n);

    pii sol;

    sol.fi = (pref[r].fi - (ll)pref[l-1].fi * p27[r-l+1].fi % mod.fi + mod.fi) % mod.fi;
    sol.se = (pref[r].se - (ll)pref[l-1].se * p27[r-l+1].se % mod.se + mod.se) % mod.se;

    return sol;
  }
};

set<pii> ss;
map<pii, int> src;
string t[5*maxn+5];
int id[maxn+5][18];

int main () {
  ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);

  string s; cin >> s;
  int n = sz(s);
  s = " " + s;

  p27[0] = pii(1, 1);
  int i, j, z;
  for (i = 1; i <= n; i++) {
    p27[i].fi = (ll)p27[i-1].fi * 27 % mod.fi;
    p27[i].se = (ll)p27[i-1].se * 27 % mod.se;
  }

  int k; cin >> k;
  for (z = 0; z < k; z++) {
    cin >> t[z];
    int m = sz(t[z]);
    t[z] = " " + t[z];
    if (m > n) continue;
    hasher T(t[z], m);
    j = 0;
    while ((1<<(j+1)) <= m) j++;
    i = 1;
    for (; j >= 0; j--) {
      if (m & (1<<j)) {
        ss.insert(T.cut(i, i+(1<<j)-1));
        i += (1<<j);
      }
    }
  }

  ///vreau sa construiesc un graf in care fiecare subsecventa a lui s cu lungime putere
  ///a lui 2 sa fie un nod propriu. mai mult, adaug o muchie intre de la un nod A la un
  ///nod B daca subsirul lui B este imediat la dreapta lui A in s.

  hasher S(s, n);
  int p2, p2_oth;

  int cnt_src = 0;
  src[pii(0,0)] = 0;
  cnt_src++;

  vector<pair<pii,int>> src_lb;
  src_lb.reserve(17 * maxn);
  for (j = 0, p2 = 1; p2 <= n; p2 <<= 1, j++) {
    for (i = 1; i <= n-p2+1; i++) {
      pii hh = S.cut(i, i+p2-1);

      id[i][j] = -1;
      if (!ss.count(hh)) {
        continue;
      }

      if (src.find(hh) == src.end()) {
        src[hh] = cnt_src;
        id[i][j] = cnt_src;
        src_lb.pb(make_pair(hh, cnt_src));
        cnt_src++;
      } else {
        id[i][j] = src[hh];
      }
    }
  }

  sort(all(src_lb));

  for (j = 0, p2 = 1; p2 <= n; p2 <<= 1, j++) {
    for (i = 1; i <= n-p2+1; i++) {
      if (id[i][j] == -1) continue;
      g[0].pb(id[i][j]);

      for (z = 0, p2_oth = 1; i+p2 + p2_oth-1 <= n; p2_oth <<= 1, z++) {
        ///vreau sa unesc [i, i+p2-1] cu [i+p2, i+p2 + p2_oth-1]
        g[id[i][j]].pb(id[i+p2][z]);
      }
    }
  }

  sort(all(g[0]));

  for (z = 0; z < k; z++) {
    int m = sz(t[z])-1;
    if (m > n) { cout << "NO\n"; continue; }
    hasher T(t[z], m);

    p2 = 1;
    while ((p2 << 1) <= m) p2 <<= 1;

    int nod = 0; ///de unde incep cautarea in graf.
    i = 1;
    while (p2 > 0) {
      if (m & p2) {
        pii hh_cautat = T.cut(i, i+p2-1);

        auto it = lower_bound(all(src_lb), make_pair(hh_cautat, -inf));
        if (it == src_lb.end() || it->fi != hh_cautat) {
          break;
        }

        int id_cautat = it->se;
        if (nod == 0) {
          if (!binary_search(all(g[nod]), id_cautat)) {
            break;
          }
        } else {
          if (find(all(g[nod]), id_cautat) == g[nod].end()) {
            break;
          }
        }

        nod = id_cautat;
        i += p2;
      }
      p2 >>= 1;
    }

    if (p2 == 0) {
      cout << "YES\n";
    } else {
      cout << "NO\n";
    }
  }

  return 0;
}
