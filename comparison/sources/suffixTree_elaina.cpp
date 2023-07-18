///https://cses.fi/problemset/hack/2103/entry/2765885/
#include "bits/stdc++.h"
 
using namespace std;
 
struct SufTree {
  const int kMod = 1.0e9 + 7;
  string s;
  int n = 0;
  vector<int> pos, len, lnk;
  vector<map<char, int>> nxt;
  int MakeNode(int pos_, int len_) {
    pos.push_back(pos_), len.push_back(len_), lnk.push_back(-1), nxt.emplace_back();
    return n++;
  }
  void Add(int& p, int& left, char c) {
    s += c, ++left;
    int last = 0;
    for ( ; left; p ? p = lnk[p] : --left) {
      auto l = static_cast<int>(s.length());
      while (left > 1 && left > len[nxt[p][s[l - left]]])
        p = nxt[p][s[l - left]], left -= len[p];
      char e = s[l - left];
      int& q = nxt[p][e];
      if (!q) {
        q = MakeNode(l - left, kMod), lnk[last] = p, last = 0;
      } else {
        char t = s[pos[q] + left - 1];
        if (t == c) {
          lnk[last] = p;
          return;
        }
        int u = MakeNode(pos[q], left - 1);
        nxt[u][c] = MakeNode(l - 1, kMod), nxt[u][t] = q;
        pos[q] += left - 1;
        if (len[q] != kMod)
          len[q] -= left - 1;
        q = u, lnk[last] = u, last = u;
      }
    }
  }
  void Initialize(const string& s_) {
    MakeNode(-1, 0);
    int p = 0, left = 0;
    for (const auto& c : s_)
      Add(p, left, c);
    Add(p, left, '$');
    s.pop_back();
  }
  int MaxPrefix(const string& s_) {
    auto n_ = static_cast<int>(s_.length());
    for (int p = 0, idx = 0; ; ) {
      if (idx == n_ || !nxt[p].count(s_[idx]))
        return idx;
      p = nxt[p][s_[idx]];
      for (int i = 0; i < len[p]; ++i) {
        if (idx == n_ || s_[idx] != s[pos[p] + i])
          return idx;
        ++idx;
      }
    }
  }
  vector<int> subtree_size;
  void Dfs(int cr) {
    subtree_size[cr] = nxt[cr].empty();
    for (auto& [c, nx] : nxt[cr])
      Dfs(nx), subtree_size[cr] += subtree_size[nx];
  }
  void GetSubtreeSizes() {
    subtree_size.resize(n);
    Dfs(0);
  }
  int NumOcc(const string& s_) {
    auto n_ = static_cast<int>(s_.length());
    int p = 0, idx = 0;
    bool done = false;
    while (true) {
      if (idx == n_ || !nxt[p].count(s_[idx]))
        break;
      p = nxt[p][s_[idx]];
      for (int i = 0; i < len[p]; ++i) {
        if (idx == n_ || s_[idx] != s[pos[p] + i]) {
          done = true;
          break;
        }
        ++idx;
      }
      if (done)
        break;
    }
    if (idx != n_)
      return 0;
    else
      return subtree_size[p];
  }
};
 
int main() {
  cin.tie(nullptr);
  ios::sync_with_stdio(false);
  string s;
  cin >> s;
  SufTree suf;
  suf.Initialize(s);
  suf.GetSubtreeSizes();
  int k;
  cin >> k;
  while (k--) {
    string p;
    //cin >> p;
    int l, r; cin >> l >> r;
    p = s.substr(l-1, r-l+1);
 
    cout << suf.NumOcc(p) << '\n';
  }
}
