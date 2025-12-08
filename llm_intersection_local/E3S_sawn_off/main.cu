#include "utils.h"

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
        thrust::transform(tmp.begin(), tmp.end(), dev_s_cuts.begin(), [] __device__ (thrust::pair<uint64_t, int> p) { return p.first; });
    }

    int q; std::cin >> q;

    thrust::device_vector<TsInfo> dev_ts_info(q);
    int m = 0;
    {
        thrust::host_vector<TsInfo> hst_ts_info(q);

        ///stream segment start_i: indexul din dictionar de la care incepe segmentul tinut momentan in memorie.
        ///citesc/tin in memorie doar ca sa calculez hash-urile pentru prefix/sufix.
        int sseg_start_i = 0, sseg_m = 0;

        std::vector<uint8_t> sseg_ts_buff(MAXM_STREAMING);
        std::vector<int> hst_set_keys_at;

        auto flush_streaming_segment = [&dev_base_pws, &sseg_m, &m, &sseg_start_i, &sseg_ts_buff, &hst_set_keys_at, &hst_ts_info](int sseg_end_i) {
            thrust::device_vector<thrust::pair<uint64_t, int>> dev_tmp_in(sseg_m), dev_tmp_out(sseg_m);
            thrust::transform(sseg_ts_buff.begin(), sseg_ts_buff.begin() + sseg_m, dev_tmp_in.begin(), [] __device__ (uint8_t ch) { return thrust::make_pair((uint64_t)ch, 1); });

            thrust::device_vector<int> dev_set_keys_at = hst_set_keys_at, dev_keys(sseg_m);
            kernel_set_keys_at<<<(hst_set_keys_at.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
                (int)hst_set_keys_at.size(), thrust::raw_pointer_cast(&dev_set_keys_at[0]), thrust::raw_pointer_cast(&dev_keys[0])
            );
            thrust::inclusive_scan(dev_keys.begin(), dev_keys.end(), dev_keys.begin());

            void *dev_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
            cub::DeviceScan::InclusiveScanByKey(
                dev_temp_storage, temp_storage_bytes,
                thrust::raw_pointer_cast(&dev_keys[0]), thrust::raw_pointer_cast(&dev_tmp_in[0]), thrust::raw_pointer_cast(&dev_tmp_out[0]),
                PrefSumModMultiples(dev_base_pws), sseg_m
            );

            cudaMalloc(&dev_temp_storage, temp_storage_bytes);
            cub::DeviceScan::InclusiveScanByKey(
                dev_temp_storage, temp_storage_bytes,
                thrust::raw_pointer_cast(&dev_keys[0]), thrust::raw_pointer_cast(&dev_tmp_in[0]), thrust::raw_pointer_cast(&dev_tmp_out[0]),
                PrefSumModMultiples(dev_base_pws), sseg_m
            );
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

            hst_set_keys_at.clear();
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

            if (sseg_m + t.size() > MAXM_STREAMING) flush_streaming_segment(i-1);
            
            hst_set_keys_at.push_back(sseg_m);
            if (pref_len < t.size()) hst_set_keys_at.push_back(sseg_m + pref_len);
            
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
        thrust::device_vector<int> dev_pref_lens(q), dev_pref_lens_out(q);
        thrust::device_vector<TsInfo> dev_ts_info_out(q);
        thrust::transform(dev_ts_info.begin(), dev_ts_info.end(), dev_pref_lens.begin(), [] __device__ (const TsInfo &t) { return t.len - t.suff_len; });

        void *dev_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(dev_temp_storage, temp_storage_bytes, thrust::raw_pointer_cast(&dev_pref_lens[0]), thrust::raw_pointer_cast(&dev_pref_lens_out[0]), thrust::raw_pointer_cast(&dev_ts_info[0]), thrust::raw_pointer_cast(&dev_ts_info_out[0]), q);
        
        cudaMalloc(&dev_temp_storage, temp_storage_bytes);
        cub::DeviceRadixSort::SortPairs(dev_temp_storage, temp_storage_bytes, thrust::raw_pointer_cast(&dev_pref_lens[0]), thrust::raw_pointer_cast(&dev_pref_lens_out[0]), thrust::raw_pointer_cast(&dev_ts_info[0]), thrust::raw_pointer_cast(&dev_ts_info_out[0]), q);
        cudaFree(dev_temp_storage);

        dev_pref_lens = dev_pref_lens_out;
        kernel_extract_ts_pref_len_offsets<<<(q + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            q, thrust::raw_pointer_cast(&dev_pref_lens[0]), thrust::raw_pointer_cast(&dev_ts_pref_offsets[0])
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
        thrust::device_vector<uint128_t> dev_keys_in(q), dev_keys_out(q);
        thrust::transform(dev_ts_info.begin(), dev_ts_info.end(), dev_keys_in.begin(), [] __device__ (const TsInfo &t) { return ((uint128_t)t.hh_p << 64) | t.hh_s; });

        dev_temp_storage = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortPairs(
            dev_temp_storage, temp_storage_bytes,
            thrust::raw_pointer_cast(&dev_keys_in[0]), thrust::raw_pointer_cast(&dev_keys_out[0]),
            thrust::raw_pointer_cast(&dev_ts_info_out[0]), thrust::raw_pointer_cast(&dev_ts_info[0]),
            q, cnt_offsets,
            thrust::raw_pointer_cast(&dev_ts_pref_offsets[0]), thrust::raw_pointer_cast(&dev_ts_pref_offsets[0]) + 1
        );

        cudaMalloc(&dev_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedRadixSort::SortPairs(
            dev_temp_storage, temp_storage_bytes,
            thrust::raw_pointer_cast(&dev_keys_in[0]), thrust::raw_pointer_cast(&dev_keys_out[0]),
            thrust::raw_pointer_cast(&dev_ts_info_out[0]), thrust::raw_pointer_cast(&dev_ts_info[0]),
            q, cnt_offsets,
            thrust::raw_pointer_cast(&dev_ts_pref_offsets[0]), thrust::raw_pointer_cast(&dev_ts_pref_offsets[0]) + 1
        );
        cudaFree(dev_temp_storage);
    }
    
    thrust::device_vector<PrefixInfo> dev_prefs(n);

    for (int off = 0; off+1 < (int)hst_ts_pref_offsets.size(); off++) {
        int offset = hst_ts_pref_offsets[off];

        TsInfo info;
        thrust::copy(dev_ts_info.begin() + offset, dev_ts_info.begin() + offset + 1, &info);

        int p2 = info.len - info.suff_len, ts_msb_l = offset, ts_msb_r = hst_ts_pref_offsets[off+1]-1;

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

                thrust::device_vector<PrefixInfo> dev_prefs_out(cnt_prefs);
                void *dev_temp_storage = nullptr;
                size_t temp_storage_bytes = 0;
                cub::DeviceRadixSort::SortPairs(
                    dev_temp_storage, temp_storage_bytes,
                    thrust::raw_pointer_cast(&dev_tmp_cat_hh_ps_in[0]), thrust::raw_pointer_cast(&dev_tmp_cat_hh_ps_out[0]),
                    thrust::raw_pointer_cast(&dev_prefs[0]), thrust::raw_pointer_cast(&dev_prefs_out[0]),
                    cnt_prefs
                );
                
                cudaMalloc(&dev_temp_storage, temp_storage_bytes);
                cub::DeviceRadixSort::SortPairs(
                    dev_temp_storage, temp_storage_bytes,
                    thrust::raw_pointer_cast(&dev_tmp_cat_hh_ps_in[0]), thrust::raw_pointer_cast(&dev_tmp_cat_hh_ps_out[0]),
                    thrust::raw_pointer_cast(&dev_prefs[0]), thrust::raw_pointer_cast(&dev_prefs_out[0]),
                    cnt_prefs
                );
                cudaFree(dev_temp_storage);

                dev_prefs = dev_prefs_out;
            }

            ///scapam de pref + shade identice.
            {
                thrust::device_vector<int> dev_levs_in(cnt_prefs, 1), dev_levs_out(cnt_prefs);

                void *dev_temp_storage = nullptr;
                size_t temp_storage_bytes = 0;
                cub::DeviceScan::InclusiveScanByKey(
                    dev_temp_storage, temp_storage_bytes,
                    thrust::raw_pointer_cast(&dev_tmp_cat_hh_ps_in[0]), thrust::raw_pointer_cast(&dev_levs_in[0]), thrust::raw_pointer_cast(&dev_levs_out[0]),
                    Uint128Equality(), cnt_prefs
                );

                cudaMalloc(&dev_temp_storage, temp_storage_bytes);
                cub::DeviceScan::InclusiveScanByKey(
                    dev_temp_storage, temp_storage_bytes,
                    thrust::raw_pointer_cast(&dev_tmp_cat_hh_ps_in[0]), thrust::raw_pointer_cast(&dev_levs_in[0]), thrust::raw_pointer_cast(&dev_levs_out[0]),
                    Uint128Equality(), cnt_prefs
                );
                cudaFree(dev_temp_storage);

                thrust::device_vector<int> dev_levs_margins(cnt_prefs);
                kernel_insert_leverage_margins<<<(cnt_prefs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
                    cnt_prefs, thrust::raw_pointer_cast(&dev_levs_out[0]), thrust::raw_pointer_cast(&dev_levs_margins[0])
                );

                thrust::inclusive_scan(dev_levs_margins.begin(), dev_levs_margins.end(), dev_levs_margins.begin());

                thrust::device_vector<PrefixInfo> dev_prefs_out(cnt_prefs);

                kernel_extract_unique_prefs<<<(cnt_prefs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
                    cnt_prefs, thrust::raw_pointer_cast(&dev_levs_margins[0]), thrust::raw_pointer_cast(&dev_prefs[0]), thrust::raw_pointer_cast(&dev_prefs_out[0])
                );

                ///noul numar de cnt_prefs = ultima valoare din cnt_prefs.
                thrust::copy(dev_levs_margins.begin() + cnt_prefs-1, dev_levs_margins.begin() + cnt_prefs, &cnt_prefs);

                thrust::copy(dev_prefs_out.begin(), dev_prefs_out.begin() + cnt_prefs, dev_prefs.begin());
            }
        }

        ///calculez raspunsul pentru elementele din ts_info[] care au MSB egal cu p2 (eg [ts_msb_l, ts_msb_r]).

        ///unde incep grupele?
        int cnt_groups;
        thrust::device_vector<int> dev_group_starts;
        {
            thrust::device_vector<int> dev_group_start_markers(cnt_prefs); ///era kernel_extract_unique_prefs inainte??
            kernel_mark_group_starts<<<(cnt_prefs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
                cnt_prefs, thrust::raw_pointer_cast(&dev_prefs[0]), thrust::raw_pointer_cast(&dev_group_start_markers[0])
            );

            thrust::inclusive_scan(dev_group_start_markers.begin(), dev_group_start_markers.end(), dev_group_start_markers.begin());

            thrust::copy(dev_group_start_markers.begin() + cnt_prefs-1, dev_group_start_markers.begin() + cnt_prefs, &cnt_groups);

            dev_group_starts.resize(cnt_groups); ///era kernel_extract_unique_prefs inainte??
            kernel_get_group_starts<<<(cnt_prefs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
                cnt_prefs, thrust::raw_pointer_cast(&dev_group_start_markers[0]), thrust::raw_pointer_cast(&dev_group_starts[0])
            );
        }

        ///ce lungimi de sufixe avem in halfway group?
        thrust::device_vector<int> dev_suff_lens(ts_msb_r-ts_msb_l+1);
        {
            thrust::transform(dev_ts_info.begin() + ts_msb_l, dev_ts_info.end() + ts_msb_r+1, dev_suff_lens.begin(), [] __device__ (const TsInfo &t) { return t.suff_len; });
            thrust::sort(dev_suff_lens.begin(), dev_suff_lens.end());
            dev_suff_lens.resize(thrust::unique(dev_suff_lens.begin(), dev_suff_lens.end()) - dev_suff_lens.begin());
        }

        kernel_solve_halfway_group<<<(cnt_groups + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            p2, thrust::raw_pointer_cast(&dev_s_cuts[0]), q,
            cnt_groups, thrust::raw_pointer_cast(&dev_group_starts[0]),
            cnt_prefs, thrust::raw_pointer_cast(&dev_prefs[0]),
            ts_msb_l, ts_msb_r, thrust::raw_pointer_cast(&dev_ts_info[0]),
            dev_suff_lens.size(), thrust::raw_pointer_cast(&dev_suff_lens[0])
        );
    }

    ///actualizare sume partiale din dev_ts_info[].count
    thrust::host_vector<int> hst_ts_count(q);
    {
        thrust::device_vector<int> dev_ts_count(q);
        thrust::transform(dev_ts_info.begin(), dev_ts_info.end(), dev_ts_count.begin(), [] __device__ (TsInfo &t) { return t.count; });
        thrust::inclusive_scan(dev_ts_count.begin(), dev_ts_count.end(), dev_ts_count.begin());

        thrust::device_vector<int> dev_ts_ind(q);
        thrust::transform(dev_ts_info.begin(), dev_ts_info.end(), dev_ts_ind.begin(), [] __device__ (TsInfo &t) { return t.ind; });

        thrust::device_vector<int> dev_ts_count_out(q), dev_ts_ind_out(q);
        void *dev_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            dev_temp_storage, temp_storage_bytes,
            thrust::raw_pointer_cast(&dev_ts_ind[0]), thrust::raw_pointer_cast(&dev_ts_ind_out[0]),
            thrust::raw_pointer_cast(&dev_ts_count[0]), thrust::raw_pointer_cast(&dev_ts_count_out[0]),
            q
        );
        
        cudaMalloc(&dev_temp_storage, temp_storage_bytes);
        cub::DeviceRadixSort::SortPairs(
            dev_temp_storage, temp_storage_bytes,
            thrust::raw_pointer_cast(&dev_ts_ind[0]), thrust::raw_pointer_cast(&dev_ts_ind_out[0]),
            thrust::raw_pointer_cast(&dev_ts_count[0]), thrust::raw_pointer_cast(&dev_ts_count_out[0]),
            q
        );
        cudaFree(dev_temp_storage);

        thrust::copy(dev_ts_count_out.begin(), dev_ts_count_out.end(), hst_ts_count.begin());
    }

    for (int cnt: hst_ts_count) std::cout << cnt << '\n';

    return 0;
}
