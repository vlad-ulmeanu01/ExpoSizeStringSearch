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
 

__global__ void kernel_compute_prefix_info(
    int cnt_prefs, int pw_msb, PrefixInfo *dev_prefs, uint64_t *dev_base_pws, uint64_t *dev_s_cuts
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cnt_prefs) return;

    dev_prefs[index].sh_start = index + pw_msb;
    int sh_end = index+pw_msb + pw_msb-2;
    dev_prefs[index].sh_end = (sh_end < n? sh_end: n-1);

    uint64_t hh_p = dev_s_cuts[index+pw_msb] - mul(dev_s_cuts[index], dev_base_pws[pw_msb]);
    dev_prefs[index].hh_p = (hh_p < 0? hh_p + M61: hh_p);

    if (dev_prefs[index].sh_start <= dev_prefs[index].sh_end) {
        uint64_t hh_s = dev_s_cuts[dev_prefs[index].sh_end + 1] - mul(dev_s_cuts[dev_prefs[index].sh_start], dev_base_pws[pw_msb-1]);
        dev_prefs[index].hh_s = (hh_s < 0? hh_s + M61: hh_s);
    } else dev_prefs[index].hh_s = 0;

    ///.lev e calculat ulterior.
}

__global__ void kernel_set_keys_at(
    int cnt_set_keys_at, int *dev_set_keys_at, int *dev_keys
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < cnt_set_keys_at) dev_keys[dev_set_keys_at[index]] = 1;
}

///folosit pentru a extrage rezultatele de dupa cub::DeviceScan::InclusiveScanByKey.
__global__ void kernel_extract_segment_finals(
    int sseg_m, int *dev_keys, thrust::pair<uint64_t, int> *dev_tmp_out, uint64_t *dev_hh_finals
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < sseg_m && (index == sseg_m-1 || dev_keys[index] != dev_keys[index+1])) dev_hh_finals[dev_keys[index] - 1] = dev_tmp_out[index].first;
}

__global__ void kernel_insert_leverage_margins(
    int cnt_prefs, int *dev_levs_out, int *dev_levs_margins
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < cnt_prefs && (index == cnt_prefs-1 || dev_levs_out[index+1] == 1)) dev_levs_margins[index] = 1;
}

__global__ void kernel_extract_unique_prefs(
    int cnt_prefs, int *dev_levs_margins, PrefixInfo *dev_prefs_in, PrefixInfo *dev_prefs_out
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < cnt_prefs && (index == cnt_prefs-1 || dev_levs_margins[index] != dev_levs_margins[index+1])) {
        dev_prefs_out[dev_levs_margins[index]-1] = dev_prefs_in[index];
        dev_prefs_out[dev_levs_margins[index]-1].lev = dev_levs_margins[index];
    }
}

///am un vector sortat cu lungimile prefixelor din ts (pref_lens). trebuie sa construiesc un vector de offset-uri, la ce index incepe urmatorul segment
///de lungimi de prefixe. prefixele sunt puteri de 2, deci sunt doar O(log ??) offset-uri diferite.
__global__ void kernel_extract_ts_pref_len_offsets(
    int q, int *pref_lens, int *pref_offsets
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < q && (index == 0 || pref_lens[index] != pref_lens[index-1])) {
        int log2_index = 0, p = pref_lens[index];
        while (p > 1) { log2_index++; p >>= 1; }
        pref_offsets[log2_index] = index;
    }
}

__global__ void kernel_mark_group_starts(
    int cnt_prefs, PrefixInfo *dev_prefs, int *dev_group_start_markers
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < cnt_prefs && (index == 0 || dev_prefs[index].hh_p != dev_prefs[index-1].hh_p)) {
        dev_group_start_markers[index] = 1;
    }
}

__global__ void kernel_get_group_starts(
    int cnt_prefs, int *dev_group_start_markers, int *dev_group_starts
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < cnt_prefs && (index == 0 || dev_group_start_markers[index] != dev_group_start_markers[index-1])) {
        dev_group_starts[dev_group_starts[index]-1] = index;
    }
}

__global__ void kernel_solve_group_child(
    int p2, uint64_t *dev_s_cuts, int q,
    PrefixInfo *dev_prefs, int pref_l, int pref_r,
    int ts_l, int ts_r, TsInfo *dev_ts_info,
    int cnt_suff_lens, int *dev_suff_lens
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pref_r - pref_l + 1) return;

    int pref_ind = pref_l + index, lev = dev_prefs[index].lev;

    ///calculam pentru dev_prefs[pref_ind].hh_s(hade) hh_s(uff) pentru toate dev_suff_lens.
    for (int i = 0; i < cnt_suff_lens; i++) {
        ///sh_start, sh_end.
        int l = dev_prefs[pref_ind].sh_start, r = l + dev_suff_lens[i] - 1;
        if (r <= dev_prefs[pref_ind].sh_end) {
            uint64_t hh_suff = dev_s_cuts[r + 1] - mul(dev_s_cuts[l], dev_base_pws[p2-1]);
            hh_suff = (hh_suff < 0? hh_suff + M61: hh_suff);

            int j = ts_r;
            for (int pas = (1<<30); pas; pas >>= 1) {
                if (j - pas >= ts_l && dev_ts_info[j-pas].hh_s >= hh_suff) j -= pas;
            }

            if (dev_ts_info[j].hh_s == hh_suff) {
                atomicAdd(dev_ts_info[j].count, lev);
                if (j == ts_r || dev_ts_info[j+1].hh_s > hh_suff) { ///am exact un singur match.
                    if (j+1 < q) atomicAdd(dev_ts_info[j+1].count, -lev);
                } else { ///trebuie sa cautam binar ultimul match.
                    int z = j;
                    for (int pas = (1<<30); pas; pas >>= 1) {
                        if (z + pas <= ts_r && dev_ts_info[z+pas].hh_s == hh_suff) z += pas;
                    }
                    if (z+1 < q) atomicAdd(dev_ts_info_out[z+1].count, -lev);
                }
            }
        }
    }
}

__global__ void kernel_solve_halfway_group(
    int p2, uint64_t *dev_s_cuts, int q,
    int cnt_groups, int *dev_group_starts,
    int cnt_prefs, PrefixInfo *dev_prefs,
    int ts_msb_l, int ts_msb_r, TsInfo *dev_ts_info,
    int cnt_suff_lens, int *dev_suff_lens
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cnt_groups) return;

    int pref_l = dev_group_starts[index], pref_r = (index == cnt_groups-1? cnt_prefs - 1: dev_group_starts[index+1] - 1);

    ///halfway group: dev_prefs[pref_l .. pref_r].hh_p e identic.
    ///DAR nu exista garantia de la grupuri pentru dev_ts_info[ts_msb_l .. ts_msb_r].hh_p (ca sunt la fel cu itv din dev_prefs).
    ///in schimb avem garantia ca 1) raspunsul pentru elem din ^^^ poate fi construit doar din dev_prefs[..].
    ///si 2) elementele din dev_ts_info sunt sortate crescator dupa (hh_pref, hh_suff).

    ///determinam grupul complet, eg limitele din dev_ts_info: [ts_l .. ts_r].
    int ts_l = ts_msb_r;
    uint64_t hh_p = dev_prefs[pref_l].hh_p;
    for (int pas = (1<<30); pas; pas >>= 1) {
        if (ts_l - pas >= ts_msb_l && dev_ts_info[ts_l - pas].hh_p >= hh_p) ts_l -= pas;
    }

    if (dev_ts_info[ts_l] != hh_p) return; ///nu exista hh_p in ts_info.
    int ts_r = ts_l;
    for (int pas = (1<<30); pas; pas >>= 1) {
        if (ts_r + pas <= ts_msb_r && dev_ts_info[ts_r + pas].hh_p == hh_p) ts_r += pas;
    }

    kernel_solve_group_child<<<(pref_r-pref_l + THREADS_PER_BLOCK) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        p2, dev_s_cuts, q, dev_prefs, pref_l, pref_r, ts_l, ts_r, dev_ts_info, cnt_suff_lens, dev_suff_lens
    );
}
