# ExpoSizeStringSearch
Exponential Length Substrings in Pattern Matching

This note describes a hash-based mass-searching algorithm, finding (count, location of first match) entries from a dictionary against a string $s$ of length $n$. The presented implementation makes use of all substrings of $s$ whose lengths are powers of $2$ to construct an offline algorithm that can, in some cases, reach a complexity of $O(n \log^2n)$ even if there are $O(n^2)$ possible matches. If there is a limit on the dictionary size $m$, then the precomputation complexity is $O(m + n \log^2n)$, and the search complexity is bounded by $O(\min (n \sqrt m \log n, n^2 \log n))$, even if it performs better in practice.
