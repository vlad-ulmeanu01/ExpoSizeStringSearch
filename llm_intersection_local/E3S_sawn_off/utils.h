#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/pair.h>

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <climits>
#include <cassert>
#include <chrono>
#include <vector>
#include <random>
#include <stack>
#include <cmath>
#include <map>

#define aaa system("read -r -p \"Press enter to continue...\" key");
#define dbg(x) std::cerr<<(#x)<<": "<<(x)<<'\n',aaa
#define dbga(x,n) std::cerr<<(#x)<<"[]: ";for(int _=0;_<n;_++)std::cerr<<x[_]<<' ';std::cerr<<'\n',aaa
#define dbgs(x) std::cerr<<(#x)<<"[stl]: ";for(auto _:x)std::cerr<<_<<' ';std::cerr<<'\n',aaa
#define dbgp(x) std::cerr<<(#x)<<": "<<x.fi<<' '<<x.se<<'\n',aaa
#define dbgsp(x) std::cerr<<(#x)<<"[stl pair]:\n";for(auto _:x)std::cerr<<_.fi<<' '<<_.se<<'\n';aaa
#define fi first
#define se second

#define uint128_t unsigned __int128

const int maxm_streaming = 1'000'000'000;

constexpr uint32_t ct229 = (1 << 29) - 1;
constexpr uint64_t M61 = (1ULL << 61) - 1, M61_2x = M61 * 2;

__host__ __device__ uint64_t mul(uint64_t a, uint64_t b);

inline uint64_t hh_add_char(uint64_t hh, uint8_t ch, uint64_t base);
inline uint64_t hh_rm_char(uint64_t hh, uint8_t ch, uint64_t base_pw);
inline uint64_t hh_roll(uint64_t hh, uint8_t ch_bye, uint8_t ch, uint64_t base, uint64_t base_pw);
inline uint64_t hh_cut(const std::vector<uint64_t> &s_cuts, const std::vector<uint64_t> &base_pws, int l, int r);

struct ModMultiplies {
    __device__ uint64_t operator()(uint64_t prev, uint64_t curr) const {
        return mul(prev, curr);
    }
};

struct PrefSumModMultiples {
    thrust::device_ptr<uint64_t> dev_base_pws;

    PrefSumModMultiples(thrust::device_vector<uint64_t>& dev_base_pws): dev_base_pws(thrust::device_pointer_cast(dev_base_pws.data())) {}

    __device__ thrust::pair<uint64_t, int> operator()(thrust::pair<uint64_t, int> prev, thrust::pair<uint64_t, int> curr) const {
        ///operatorul trebuie se fie asociativ pentru versiunea curenta de inclusive_scan. fi = hash, se = lungime.

        uint64_t hh = mul(prev.first, dev_base_pws[curr.second]) + curr.first;
        hh = (hh >= M61? hh - M61: hh);

        return thrust::make_pair(hh, prev.second + curr.second);
    }
};
