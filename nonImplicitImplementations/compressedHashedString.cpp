#include <bits/stdc++.h>
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

using namespace std;

class CompressedHashedString {
private:
  const pii mod = pii(1000000007, 1000000009);
  const pii inv26 = pii(576923081, 807692315);

  int lgput(int a, int b, int m) {
    int ans = 1, p2 = 1;
    while (b) {
      if (b & p2) {
        ans = (ll)ans * a % m;
        b ^= p2;
      }
      p2 <<= 1;
      a = (ll)a * a % m;
    }
    return ans;
  }

  string s;
  int indNow;
  vector<pair<pii, int>> pPH; ///= partial prefix hashes. fi = hash, se = current length of prefix.
  vector<char> chars;

  int getInt(string &s) {
    int ans = 0;
    while (indNow < sz(s) && s[indNow] >= '0' && s[indNow] <= '9') {
      ans = ans * 10 + s[indNow] - '0';
      indNow++;
    }
    return ans;
  }

  ///H(s[1..l]) = ?
  pii getHhPrefix(int l) {
    int i, pas; ///i = biggest index for which pPH[i]'s length <= l.
    for (pas = (1<<30), i = 0; pas > 0; pas /= 2)
      if (i+pas < sz(pPH) && pPH[i+pas].se <= l) i += pas;

    if (pPH[i].se == l) {
      return pPH[i].fi;
    }

    int len = l - pPH[i].se, p27;
    char ch = chars[i+1];
    pii hh = pPH[i].fi;

    p27 = lgput(27, len, mod.fi);
    hh.fi = ((ll)hh.fi * p27 % mod.fi + (ll)(ch-'a'+1) * (p27 + mod.fi-1) % mod.fi * inv26.fi % mod.fi) % mod.fi;
    p27 = lgput(27, len, mod.se);
    hh.se = ((ll)hh.se * p27 % mod.se + (ll)(ch-'a'+1) * (p27 + mod.se-1) % mod.se * inv26.se % mod.se) % mod.se;

    return hh;
  }

public:
  pii cut(int l, int r) {
    pii hhR = getHhPrefix(r), hhL = getHhPrefix(l-1);

    hhL.fi = (ll)hhL.fi * lgput(27, r-l+1, mod.fi) % mod.fi;
    hhL.se = (ll)hhL.se * lgput(27, r-l+1, mod.se) % mod.se;

    return pii(
      ((ll)hhR.fi - hhL.fi + mod.fi) % mod.fi,
      ((ll)hhR.se - hhL.se + mod.se) % mod.se
    );
  }

  int totLen;
  void init(string s_) {
    s = " " + s_;
    indNow = 1;
    totLen = 0;
    pPH.clear();
    chars.clear();

    pPH.pb(make_pair(pii(0, 0), 0));
    chars.pb(' ');
    int len, p27;
    pii hh; char ch;
    while (indNow < sz(s)) {
      len = max(1, getInt(s));
      ch = s[indNow++];

      p27 = lgput(27, len, mod.fi);
      hh.fi = ((ll)pPH.back().fi.fi * p27 + (ll)(ch-'a'+1) * (p27 + mod.fi-1) % mod.fi * inv26.fi % mod.fi) % mod.fi;
      p27 = lgput(27, len, mod.se);
      hh.se = ((ll)pPH.back().fi.se * p27 + (ll)(ch-'a'+1) * (p27 + mod.se-1) % mod.se * inv26.se % mod.se) % mod.se;

      totLen += len;
      pPH.pb(make_pair(hh, totLen));
      chars.pb(ch);
    }
  }
};

CompressedHashedString CHS;

int main() {
  ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);

  string s; cin >> s;

  CHS.init(s);
  dbgp(CHS.cut(1, 8));
  dbgp(CHS.cut(9, 12));
  dbgp(CHS.cut(13, 14));

  return 0;
}
