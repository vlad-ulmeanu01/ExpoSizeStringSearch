#include "utils.h"

///a = a * b % M61.
__host__ __device__ uint64_t mul(uint64_t a, uint64_t b) {
    uint64_t a_hi = a >> 32, a_lo = (uint32_t)a, b_hi = b >> 32, b_lo = (uint32_t)b, ans = 0, tmp = 0;
 
    tmp = a_hi * b_lo + a_lo * b_hi;
    tmp = ((tmp & ct229) << 32) + (tmp >> 29);
    tmp += (a_hi * b_hi) << 3;
 
    ans = (tmp >> 61) + (tmp & M61);
    tmp = a_lo * b_lo;
 
    ans += (tmp >> 61) + (tmp & M61);
    ans = (ans >= M61_2x? ans - M61_2x: (ans >= M61? ans - M61: ans));
    return ans;
}
 
inline uint64_t hh_add_char(uint64_t hh, uint8_t ch, uint64_t base) {
    hh = mul(hh, base) + ch;
    return (hh >= M61? hh - M61: hh);
}
 
inline uint64_t hh_rm_char(uint64_t hh, uint8_t ch, uint64_t base_pw) {
    uint64_t sub = mul(ch, base_pw);
    return (hh >= sub? hh - sub: hh + M61 - sub);
}
 
///TODO sunt putine combinatii de ch_bye si ch, poti sa faci ceva cu ele?
inline uint64_t hh_roll(uint64_t hh, uint8_t ch_bye, uint8_t ch, uint64_t base, uint64_t base_pw) {    
    hh = hh_rm_char(hh, ch_bye, base_pw);
    return hh_add_char(hh, ch, base);
}
 
///s_cuts[0 .. n]. 0 <= l <= r < n.
inline uint64_t hh_cut(const std::vector<uint64_t> &s_cuts, const std::vector<uint64_t> &base_pws, int l, int r) {
    uint64_t sub = mul(s_cuts[l], base_pws[r-l+1]);
    return (s_cuts[r+1] >= sub? s_cuts[r+1] - sub: s_cuts[r+1] + M61 - sub);
}
