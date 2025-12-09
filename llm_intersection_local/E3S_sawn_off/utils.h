#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/pair.h>
#include <cub/cub.cuh>

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

using uint128_t = unsigned __int128;

namespace cub {
    template <>
    struct Traits<uint128_t> {
        enum {
            BITS = sizeof(uint128_t) * CHAR_BIT,
            CATEGORY = 0,
            IsFloatingPoint = 0
        };

        static const uint128_t LOWEST_KEY = 0;
        static const uint128_t MAX_KEY = ~(uint128_t)0;

        typedef uint128_t UnsignedBits;
        typedef __int128 SignedBits;
        typedef uint128_t RawType;
        typedef uint128_t TwiddleIn;
        typedef uint128_t TwiddleOut;
    };
}

struct Uint128Equality {
    __host__ __device__ bool operator()(uint128_t const &a, uint128_t const &b) const {
        return a == b;
    }
};

const int THREADS_PER_BLOCK = 256;
const int MAXM_STREAMING = 1'000'000; ///1'000'000'000

constexpr uint32_t ct229 = (1 << 29) - 1;
constexpr uint64_t M61 = (1ULL << 61) - 1, M61_2x = M61 * 2;


///folosit pentru bucati puteri de 2 din s.
struct PrefixInfo {
    uint64_t hh_p, hh_s; ///hash pref (primele pw_msb caractere), shade (urmatoarele pw_msb - 1 caractere).
    int sh_start, sh_end; ///unde incepe, unde se termina (inclusiv) shade-ul prefixului.
    int lev; ///pot sa am mai multe intrari de PrefixInfo indentice (acelasi pref_hh, acelasi shade), le compresez <=> compresie DAG.
};

struct TsInfo {
    uint64_t hh_p, hh_s; ///hash pref (primele pw_msb caractere), suff (restul).
    int len, suff_len; ///lungime, lungimea sufixului. tin minte si lungimea totala ptc calculez hh_ps imediat cum primesc un t, nu-l tin pe tot in memorie.
    int ind; ///indexul in ts.
    int count; ///raspunsul.
};

__host__ __device__ uint64_t mul(uint64_t a, uint64_t b);

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

__global__ void kernel_compute_prefix_info(int n, int cnt_prefs, int pw_msb, PrefixInfo *dev_prefs, uint64_t *dev_base_pws, uint64_t *dev_s_cuts);

__global__ void kernel_set_keys_at(int cnt_set_keys_at, int *dev_set_keys_at, int *dev_keys);

__global__ void kernel_extract_segment_finals(int sseg_m, int *dev_keys, thrust::pair<uint64_t, int> *dev_tmp_out, uint64_t *dev_hh_finals);

__global__ void kernel_insert_leverage_margins(int cnt_prefs, int *dev_levs_out, int *dev_levs_margins);

__global__ void kernel_extract_unique_prefs(int cnt_prefs, int *dev_levs_margins, PrefixInfo *dev_prefs_in, PrefixInfo *dev_prefs_out);

__global__ void kernel_extract_ts_pref_len_offsets(int q, int *pref_lens, int *pref_offsets);

__global__ void kernel_mark_group_starts(int cnt_prefs, PrefixInfo *dev_prefs, int *dev_group_start_markers);

__global__ void kernel_get_group_starts(int cnt_prefs, int *dev_group_start_markers, int *dev_group_starts);

__global__ void kernel_solve_group_child(
    int q, int p2, uint64_t *dev_base_pws, uint64_t *dev_s_cuts,
    PrefixInfo *dev_prefs, int pref_l, int pref_r,
    int ts_l, int ts_r, TsInfo *dev_ts_info,
    int cnt_suff_lens, int *dev_suff_lens
);

__global__ void kernel_solve_halfway_group(
    int q, int p2, uint64_t *dev_base_pws, uint64_t *dev_s_cuts,
    int cnt_groups, int *dev_group_starts,
    int cnt_prefs, PrefixInfo *dev_prefs,
    int ts_msb_l, int ts_msb_r, TsInfo *dev_ts_info,
    int cnt_suff_lens, int *dev_suff_lens
);
