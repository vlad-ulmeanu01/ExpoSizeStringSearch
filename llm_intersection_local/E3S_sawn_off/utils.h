#pragma once
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/generate.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/pair.h>
#include <cub/cub.cuh>

#include <parquet/arrow/reader.h>
#include <arrow/chunked_array.h>
#include <arrow/pretty_print.h>
#include <arrow/io/api.h>
#include <arrow/array.h>
#include <arrow/type.h>

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <typeinfo>
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
#define NAME(x) (#x)
#define DBG(x) std::cerr<<(#x)<<": "<<(x)<<'\n'<<std::flush;
#define DBGA(x,n) { std::cerr<<(#x)<<"[]: "; for(int _=0;_<n;_++) std::cerr<<x[_]<<' '; std::cerr<<'\n'<<std::flush; }
#define DBGS(x) { std::cerr<<(#x)<<"[stl]: "; for(auto _: x) std::cerr<<_<<' '; std::cerr<<'\n'<<std::flush; }
#define DBGP(x) std::cerr<<(#x)<<": "<<x.first<<' '<<x.second<<'\n'<<std::flush;
#define DBGSP(x) { std::cerr<<(#x)<<"[stl pair]:\n"; for(auto _: x) std::cerr<<_.first<<' '<<_.second<<'\n'<<std::flush; }

#define TIMER_START auto start = std::chrono::steady_clock::now(), stop = std::chrono::steady_clock::now();
#define TIMER_SPLIT(x) stop = std::chrono::steady_clock::now(); dbg_gpu_mem(); std::cerr << "(timer split) " << x << ": " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() * 1e-6 << " s \n"; start = std::chrono::steady_clock::now();

using uint128_t = unsigned __int128;

constexpr bool DEBUG_FLAG = false;
constexpr uint64_t DEBUG_SEED = 8701438702UL;

const bool RUN_LOCAL = false;

// const int CNT_PARQUET_FILES = 4; ///daca este < 0 => format CSES.
const std::string PARQUET_DIR = (RUN_LOCAL?
    "/home/vlad/Documents/SublimeMerge/ExpoSizeStringSearch/llm_intersection_local/intersection_test_files/the_pile_deduplicated/":
    "/export/home/acs/stud/v/vlad_adrian.ulmeanu/E3S_local/llm_copyright/the_pile_deduplicated/"
);

// const int CNT_ATTACK_FILES = 1; ///trebuie sa fie >= 1.
const int BATCH_FILE_MAX_SIZE = (50 << 20);
const std::string ATTACK_DIR = (RUN_LOCAL?
    "/home/vlad/Documents/SublimeMerge/ExpoSizeStringSearch/llm_intersection_local/intersection_test_files/outputs_pythia_batched/":
    "/export/home/acs/stud/v/vlad_adrian.ulmeanu/E3S_local/llm_copyright/outputs_pythia_batched/"
);

constexpr int PARQUET_BYTES_PER_READ = 100;

constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAXM_STREAMING = (RUN_LOCAL? 1'000'000: 1'000'000); ///1'000'000'000

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


template<typename T>
void dbgd(thrust::device_vector<T> &d) {
    thrust::host_vector<T> h = d;
    std::cerr << " [device vector of size " << h.size() << "]:\n";
    for (int i = 0; i < (int)h.size(); i++) {
        if constexpr (std::is_same_v<T, TsInfo>) {
            std::cerr << "[" << i << "]: " << "len(" << h[i].len << "), suff_len(" << h[i].suff_len << "), ind(" << h[i].ind <<
                         "), count(" << h[i].count << "), hh_p(" << h[i].hh_p << "), hh_s(" << h[i].hh_s << ")\n";
        } else if constexpr (std::is_same_v<T, PrefixInfo>) {
            std::cerr << "[" << i << "]: " << "sh_start(" << h[i].sh_start << "), sh_end(" << h[i].sh_end << "), lev(" << h[i].lev <<
                         "), hh_p(" << h[i].hh_p << "), hh_s(" << h[i].hh_s << ")\n";
        } else if constexpr (std::is_same_v<T, uint128_t>) {
            std::cerr << "<" << (uint64_t)(h[i] >> 64) << ", " << (uint64_t)h[i] << ">" << (i+1 < (int)h.size()? ", ": "");
        } else {
            std::cerr << h[i] << ' ';
        }
    }
    std::cerr << '\n';
}
#define DBGD(type, d) { std::cerr << NAME(d); dbgd<type>(d); }

struct IntSum {
    __host__ __device__ int operator()(int const &a, int const &b) const {
        return a + b;
    }
};

struct Uint128Equality {
    __host__ __device__ bool operator()(uint128_t const &a, uint128_t const &b) const {
        return a == b;
    }
};

void dbg_gpu_mem();

std::string pad_parquet_fname(int ind);

uint64_t get_filesize(std::string fname);

__host__ __device__ int get_msb(int x);

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

__global__ void kernel_extract_unique_prefs(int cnt_prefs, int *dev_levs_out, int *dev_levs_margins, PrefixInfo *dev_prefs_in, PrefixInfo *dev_prefs_out);

__global__ void kernel_extract_ts_pref_len_offsets(int q, int *dev_pref_lens, int *dev_pref_offsets);

__global__ void kernel_mark_group_starts(int cnt_prefs, PrefixInfo *dev_prefs, int *dev_group_start_markers);

__global__ void kernel_get_group_starts(int cnt_prefs, int *dev_group_start_markers, int *dev_group_starts);

__global__ void kernel_halfway_group_get_ts_ends(
    int cnt_groups, int *dev_group_starts,
    int cnt_prefs, PrefixInfo *dev_prefs,
    int ts_msb_l, int ts_msb_r, TsInfo *dev_ts_info, thrust::pair<int, int> *dev_group_ts_ends
);

__global__ void kernel_solve_groups(
    int q, uint64_t *dev_base_pws, uint64_t *dev_s_cuts,
    int cnt_groups, int *dev_group_starts,
    int cnt_prefs, PrefixInfo *dev_prefs,
    int ts_msb_l, int ts_msb_r, TsInfo *dev_ts_info, thrust::pair<int, int> *dev_group_ts_ends,
    int cnt_suff_lens, int *dev_suff_lens
);
