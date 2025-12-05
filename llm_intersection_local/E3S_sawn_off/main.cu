#include "utils.h"


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


__global__ kernel_compute_prefix_info(int cnt_prefs, int pw_msb, PrefixInfo *dev_prefs, uint64_t *dev_base_pws, uint64_t *dev_s_cuts) {
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
    int ts_ind_l, int ts_ind_r, TsInfo *dev_ts_info,
    int cnt_suff_lens, int *dev_suff_lens
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cnt_groups) return;

    int pref_l = dev_group_starts[index], pref_r = (index == cnt_groups-1? cnt_prefs - 1: dev_group_starts[index+1] - 1);

    ///halfway group: dev_prefs[pref_l .. pref_r].hh_p e identic.
    ///DAR nu exista garantia de la grupuri pentru dev_ts_info[ts_ind_l .. ts_ind_r].hh_p (ca sunt la fel cu itv din dev_prefs).
    ///in schimb avem garantia ca 1) raspunsul pentru elem din ^^^ poate fi construit doar din dev_prefs[..].
    ///si 2) elementele din dev_ts_info sunt sortate crescator dupa (hh_pref, hh_suff).

    ///determinam grupul complet, eg limitele din dev_ts_info: [ts_l .. ts_r].
    int ts_l = ts_ind_r;
    uint64_t hh_p = dev_prefs[pref_l].hh_p;
    for (int pas = (1<<30); pas; pas >>= 1) {
        if (ts_l - pas >= ts_ind_l && dev_ts_info[ts_l - pas].hh_p >= hh_p) ts_l -= pas;
    }

    if (dev_ts_info[ts_l] != hh_p) return; ///nu exista hh_p in ts_info.
    int ts_r = ts_l;
    for (int pas = (1<<30); pas; pas >>= 1) {
        if (ts_r + pas <= ts_ind_r && dev_ts_info[ts_r + pas].hh_p == hh_p) ts_r += pas;
    }

    kernel_solve_group_child<<<(pref_r-pref_l + THREADS_PER_BLOCK) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        p2, dev_s_cuts, q, dev_prefs, pref_l, pref_r, ts_l, ts_r, dev_ts_info, cnt_suff_lens, dev_suff_lens
    );
}

__global__ void kernel_get_ind_count(
    int q, TsInfo *dev_ts_info, int *dev_ts_count, thrust::pair<int, int> *dev_ts_ind_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= q) return;

    dev_ts_ind_count[index] = thrust::make_pair(dev_ts_info[index].ind, dev_ts_count[index]);
}

int main() {
    std::ios::sync_with_stdio(false); std::cin.tie(0); std::cout.tie(0);

    std::string s; std::cin >> s;
    int n = s.size();

    std::mt19937_64 mt(time(NULL));
    uint64_t base = std::uniform_int_distribution<uint64_t>(257, M61 - 1)(mt);

    ///calcul dev_base_pws.
    thrust::device_vector<uint64_t> dev_base_pws(n+1);
    {
        thrust::device_vector<uint64_t> tmp(n+1, base); tmp[0] = 1;
        thrust::inclusive_scan(tmp.begin(), tmp.end(), dev_base_pws.begin(), ModMultiplies());        
    }

    ///calcul dev_s_cuts.
    thrust::device_vector<uint64_t> dev_s_cuts(n+1);
    {
        thrust::device_vector<thrust::pair<uint64_t, int>> tmp(n+1);
        {
            thrust::device_vector<uint8_t> tmp_s(n);
            thrust::copy(s.begin(), s.end(), tmp_s.begin());
            thrust::transform(tmp_s.begin(), tmp_s.end(), tmp.begin() + 1, [] __device__ (uint8_t ch) { return thrust::make_pair((uint64_t)ch, 1); });
        }

        thrust::inclusive_scan(tmp.begin(), tmp.end(), tmp.begin(), PrefSumModMultiples(dev_base_pws));
        thrust::transform(tmp_out.begin(), tmp_out.end(), dev_s_cuts.begin(), [] __device__ (thrust::pair<uint64_t, int> p) { return p.first; });
    }

    int q; std::cin >> q;

    thrust::device_vector<TsInfo> dev_ts_info(q);
    int m = 0;
    {
        thrust::host_vector<TsInfo> hst_ts_info(q);

        ///stream segment start_i: indexul din dictionar de la care incepe segmentul tinut momentan in memorie.
        ///citesc/tin in memorie doar ca sa calculez hash-urile pentru prefix/sufix.
        int sseg_start_i = 0, sseg_m = 0;

        std::vector<uint8_t> sseg_ts_buff(maxm_streaming);
        std::vector<int> set_keys_at;

        auto flush_streaming_segment = [&dev_base_pws, &sseg_m, &sseg_start_i, &sseg_ts_buff, &set_keys_at, &hst_ts_info](int sseg_end_i) {
            thrust::device_vector<thrust::pair<uint64_t, int>> dev_tmp_in(sseg_m), dev_tmp_out(sseg_m);
            thrust::transform(sseg_ts_buff.begin(), sseg_ts_buff.begin() + sseg_m, dev_tmp_in.begin(), [] __device__ (uint8_t ch) { return thrust::make_pair((uint64_t)ch, 1); });

            for (int z: set_keys_at) dev_keys[z] = 1;
            thrust::inclusive_scan(dev_keys.begin(), dev_keys.end(), dev_keys.begin());

            void *dev_temp_storage = nullptr;
            size_t temp_storage_bytes;
            cub::DeviceScan::InclusiveScanByKey(dev_temp_storage, temp_storage_bytes, &dev_keys[0], &dev_tmp_in[0], &dev_tmp_out[0], PrefSumModMultiples(dev_base_pws), sseg_m);

            cudaMalloc(&dev_temp_storage, temp_storage_bytes);
            cub::DeviceScan::InclusiveScanByKey(dev_temp_storage, temp_storage_bytes, &dev_keys[0], &dev_tmp_in[0], &dev_tmp_out[0], PrefSumModMultiples(dev_base_pws), sseg_m);
            cudaFree(dev_temp_storage);

            thrust::device_vector<uint64_t> dev_hh_finals(sseg_end_i - sseg_start_i + 1);
            kernel_extract_segment_finals<<<(sseg_m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
                sseg_m, thrust::raw_pointer_cast(&dev_keys[0]), thrust::raw_pointer_cast(&dev_tmp_out[0]), thrust::raw_pointer_cast(&dev_hh_finals[0])
            );

            thrust::host_vector<uint64_t> hst_hh_finals = dev_hh_finals;
            for (int j = sseg_start_i, z = 0; j <= sseg_end_i; j++) {
                hst_ts_info[j].hh_p = hst_hh_finals[z++];
                if (hst_ts_info[j].suff_len > 0) hst_ts_info[j].hh_s = hst_hh_finals[z++];
            }

            set_keys_at.clear();
            sseg_start_i = sseg_end_i+1;
            m += sseg_m;
            sseg_m = 0;
        };

        std::string t;
        for (int i = 0; i < q; i++) {
            std::cin >> t;

            int pref_len = 1 << (31 - __builtin_clz(t.size()));
            hst_ts_info[i].suff_len = (int)t.size() - pref_len;
            hst_ts_info[i].len = t.size();
            hst_ts_info[i].ind = i;
            hst_ts_info[i].count = 0;

            if (sseg_m + t.size() > maxm_streaming) flush_streaming_segment(i-1);
            
            set_keys_at.push_back(sseg_m);
            if (pref_len < t.size()) set_keys_at.push_back(sseg_m + pref_len);
            
            std::copy(t.begin(), t.end(), sseg_ts_buff.begin() + sseg_m);
            sseg_m += t.size();
        }

        flush_streaming_segment(q-1);

        dev_ts_info = hst_ts_info;
    }

    // std::sort(ts_tmp.begin(), ts_tmp.begin() + k, [](const TsInfo &a, const TsInfo &b) {
    //     if (a.hh_ps[0] != b.hh_ps[0]) return a.hh_ps[0] < b.hh_ps[0]; ///intai acelasi prefix ca sa putem face grupurile.
    //     if (a.suff_len != b.suff_len) return a.suff_len < b.suff_len; ///apoi aceeasi lungime. pentru aceeasi lungime putem calcula intr-o sg trecere prin partea cealalta a grupului. (aveam "<"..)
    //     return a.hh_ps[1] < b.hh_ps[1]; ///iar in final dupa sufix. daca si sufixul e identic, pot refolosi rezultatul de dinainte.
    // });
    thrust::device_vector<int> dev_ts_pref_offsets(30, -1);
    thrust::host_vector<int> hst_ts_pref_offsets;
    {
        int cnt_offsets = 0;

        ///intai sortam dupa pref_len si obtinem offset-urile.
        thrust::device_vector<int> pref_lens(q), pref_lens_out(q);
        thrust::device_vector<TsInfo> dev_ts_info_out(q);
        thrust::transform(dev_ts_info.begin(), dev_ts_info.end(), pref_lens.begin(), [] __device__ (const TsInfo &t) { return t.len - t.suff_len; });

        void *dev_temp_storage = nullptr;
        size_t temp_storage_bytes;
        cub::DeviceRadixSort::SortPairs(dev_temp_storage, temp_storage_bytes, &pref_lens[0], &pref_lens_out[0], &dev_ts_info[0], &dev_ts_info_out[0], q);
        
        cudaMalloc(&dev_temp_storage, temp_storage_bytes);
        cub::DeviceRadixSort::SortPairs(dev_temp_storage, temp_storage_bytes, &pref_lens[0], &pref_lens_out[0], &dev_ts_info[0], &dev_ts_info_out[0], q);
        cudaFree(dev_temp_storage);

        pref_lens = pref_lens_out;
        kernel_extract_ts_pref_len_offsets<<<(cnt_prefs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            q, thrust::raw_pointer_cast(&pref_lens[0]), thrust::raw_pointer_cast(&dev_ts_pref_offsets[0])
        );

        thrust::host_vector<int> tmp_off1 = dev_ts_pref_offsets, tmp_off2(dev_ts_pref_offsets.size());
        for (int i = 0; i < (int)tmp_off1.size(); i++) {
            if (tmp_off1[i] != -1) tmp_off2[cnt_offsets++] = tmp_off1[i];
        }
        tmp_off2[cnt_offsets] = q; ///trebuie adaugat si sfarsitul ultimului segment.

        tmp_off2.resize(cnt_offsets + 1);
        dev_ts_pref_offsets = tmp_off2;
        hst_ts_pref_offsets = tmp_off2;

        ///offsets = pref len diferite, value = TsInfo(..), key = prefix & sufix combinat in uint128_t.
        thrust::device_vector<uint128_t> keys_in(q), keys_out(q);
        thrust::transform(dev_ts_info.begin(), dev_ts_info.end(), keys_in.begin(), [] __device__ (const TsInfo &t) { return ((uint128_t)p.hh_p << 64) | p.hh_s; });

        void *dev_temp_storage = nullptr;
        size_t temp_storage_bytes;
        cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, &keys_in[0], &keys_out[0], &dev_ts_info_out[0], &dev_ts_info[0], q, cnt_offsets, &dev_ts_pref_offsets[0], &dev_ts_pref_offsets[0] + 1);

        cudaMalloc(&dev_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, &keys_in[0], &keys_out[0], &dev_ts_info_out[0], &dev_ts_info[0], q, cnt_offsets, &dev_ts_pref_offsets[0], &dev_ts_pref_offsets[0] + 1);
        cudaFree(dev_temp_storage);
    }
    
    std::device_vector<PrefixInfo> dev_prefs(n);

    for (int off = 0; off+1 < (int)hst_ts_pref_offsets.size(); off++) {
        int offset = hst_ts_pref_offsets[off], p2 = dev_ts_info[offset].len - dev_ts_info[offset].suff_len, ts_ind_l = offset, ts_ind_r = hst_ts_pref_offsets[off+1]-1;

        ///generez toate subsecv de lungime p2 din s, shade-urile lor, tin minte locatiile shade-urilor.
        int cnt_prefs = n+1 - p2;
        kernel_compute_prefix_info<<<(cnt_prefs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            cnt_prefs, p2, thrust::raw_pointer_cast(&dev_prefs[0]), thrust::raw_pointer_cast(&dev_base_pws[0]), thrust::raw_pointer_cast(&dev_s_cuts[0])
        );

        {
            thrust::device_vector<uint128_t> dev_tmp_cat_hh_ps_in(cnt_prefs);
            thrust::transform(
                dev_prefs.begin(), dev_prefs.begin() + cnt_prefs, dev_tmp_cat_hh_ps_in.begin(),
                [] __device__ (const PrefixInfo &p) { return ((uint128_t)p.hh_p << 64) | p.hh_s; }
            );

            // std::sort(prefs.begin(), prefs.begin() + cnt_prefs, [](const PrefixInfo &a, const PrefixInfo &b) { return a.hh_ps < b.hh_ps; });
            {
                thrust::device_vector<uint128_t> dev_tmp_cat_hh_ps_out(cnt_prefs);

                std::device_vector<PrefixInfo> dev_prefs_out(cnt_prefs);
                void *dev_temp_storage = nullptr;
                size_t temp_storage_bytes;
                cub::DeviceRadixSort::SortPairs(dev_temp_storage, temp_storage_bytes, &dev_tmp_cat_hh_ps_in[0], &dev_tmp_cat_hh_ps_out[0], &dev_prefs[0], &dev_prefs_out[0], cnt_prefs);
                
                cudaMalloc(&dev_temp_storage, temp_storage_bytes);
                cub::DeviceRadixSort::SortPairs(dev_temp_storage, temp_storage_bytes, &dev_tmp_cat_hh_ps_in[0], &dev_tmp_cat_hh_ps_out[0], &dev_prefs[0], &dev_prefs_out[0], cnt_prefs);
                cudaFree(dev_temp_storage);

                dev_prefs = dev_prefs_out;
            }

            ///scapam de pref + shade identice.
            {
                thrust::device_vector<int> dev_levs_in(cnt_prefs, 1), dev_levs_out(cnt_prefs);

                void *dev_temp_storage = nullptr;
                size_t temp_storage_bytes;
                cub::DeviceScan::InclusiveScanByKey(dev_temp_storage, temp_storage_bytes, &dev_tmp_cat_hh_ps_in[0], &dev_levs_in[0], &dev_levs_out[0], cnt_prefs);

                cudaMalloc(&dev_temp_storage, temp_storage_bytes);
                cub::DeviceScan::InclusiveScanByKey(dev_temp_storage, temp_storage_bytes, &dev_tmp_cat_hh_ps_in[0], &dev_levs_in[0], &dev_levs_out[0], cnt_prefs);
                cudaFree(dev_temp_storage);

                thrust::device_vector<int> dev_levs_margins(cnt_prefs);
                kernel_insert_leverage_margins<<<(cnt_prefs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
                    cnt_prefs, thrust::raw_pointer_cast(&dev_levs_out[0]), thrust::raw_pointer_cast(&dev_levs_margins[0])
                );

                thrust::inclusive_scan(dev_levs_margins.begin(), dev_levs_margins.end(), dev_levs_margins.begin());

                std::device_vector<PrefixInfo> dev_prefs_out(cnt_prefs);

                kernel_extract_unique_prefs<<<(cnt_prefs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
                    cnt_prefs, thrust::raw_pointer_cast(&dev_levs_margins[0]), thrust::raw_pointer_cast(&dev_prefs[0]), thrust::raw_pointer_cast(&dev_prefs_out[0])
                );

                ///noul numar de cnt_prefs = ultima valoare din cnt_prefs.
                thrust::copy(dev_levs_margins.begin() + cnt_prefs-1, dev_levs_margins.begin() + cnt_prefs, &cnt_prefs);

                thrust::copy(dev_prefs_out.begin(), dev_prefs_out.begin() + cnt_prefs, dev_prefs.begin());
            }
        }

        ///calculez raspunsul pentru elementele din ts_info[] care au MSB egal cu p2 (eg [ts_ind_l, ts_ind_r]).

        ///unde incep grupele?
        int cnt_groups;
        thrust::device_vector<int> dev_group_starts;
        {
            thrust::device_vector<int> dev_group_start_markers(cnt_prefs);
            kernel_extract_unique_prefs<<<(cnt_prefs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
                cnt_prefs, dev_prefs, dev_group_start_markers
            );

            thrust::inclusive_scan(group_start_markers.begin(), group_start_markers.end(), group_start_markers.begin());

            thrust::copy(dev_group_start_markers.begin() + cnt_prefs-1, dev_group_start_markers.begin() + cnt_prefs, &cnt_groups);

            dev_group_starts.resize(cnt_groups);
            kernel_extract_unique_prefs<<<(cnt_prefs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
                cnt_prefs, thrust::raw_pointer_cast(&dev_group_start_markers[0]), thrust::raw_pointer_cast(&dev_group_starts[0])
            );
        }

        ///ce lungimi de sufixe avem in halfway group?
        thrust::device_vector<int> dev_suff_lens(ts_ind_r-ts_ind_l+1);
        {
            thrust::transform(dev_ts_info.begin() + ts_ind_l, dev_ts_info.end() + ts_ind_r+1, dev_suff_lens.begin(), [] __device__ (const TsInfo &t) { return t.suff_len; });
            thrust::sort(dev_suff_lens.begin(), dev_suff_lens.end());
            dev_suff_lens.resize(thrust::unique(dev_suff_lens.begin(), dev_suff_lens.end()));
        }

        kernel_solve_halfway_group<<<(cnt_groups + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            p2, thrust::raw_pointer_cast(&dev_s_cuts[0]), q,
            cnt_groups, thrust::raw_pointer_cast(&dev_group_starts[0]),
            cnt_prefs, thrust::raw_pointer_cast(&dev_prefs[0]),
            ts_ind_l, ts_ind_r, thrust::raw_pointer_cast(&dev_ts_info[0]),
            cnt_suff_lens, thrust::raw_pointer_cast(&dev_suff_lens[0])
        );
    }

    ///actualizare sume partiale din dev_ts_info[].count
    thrust::host_vector<int> hst_ts_count(q);
    {
        thrust::device_vector<int> dev_ts_count(q);
        thrust::transform(dev_ts_info.begin(), dev_ts_info.end(), dev_ts_count.begin(), [] __device__ (TsInfo &t) { return t.count; });
        thrust::inclusive_scan(dev_ts_count.begin(), dev_ts_count.end());

        thrust::device_vector<int> dev_ts_ind(q);
        thrust::transform(dev_ts_info.begin(), dev_ts_info.end(), dev_ts_ind_count.begin(), [] __device__ (TsInfo &t) { return t.ind; });

        thrust::device_vector<int> dev_ts_count_out(q), dev_ts_ind_out(q);
        void *dev_temp_storage = nullptr;
        size_t temp_storage_bytes;
        cub::DeviceRadixSort::SortPairs(dev_temp_storage, temp_storage_bytes, &dev_ts_ind[0], &dev_ts_ind_out[0], &dev_ts_count[0], &dev_ts_count_out[0], q);
        
        cudaMalloc(&dev_temp_storage, temp_storage_bytes);
        cub::DeviceRadixSort::SortPairs(dev_temp_storage, temp_storage_bytes, &dev_ts_ind[0], &dev_ts_ind_out[0], &dev_ts_count[0], &dev_ts_count_out[0], q);
        cudaFree(dev_temp_storage);

        thrust::copy(dev_ts_count_out.begin(), dev_ts_count_out.end(), hst_ts_count.begin());
    }

    for (int cnt: hst_ts_count) std::cout << cnt << '\n';

    return 0;
}
