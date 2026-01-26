#pragma once
#include "utils.h"
#include "readers.h"

struct E3S_sawnoff {
    std::random_device rd;
    std::mt19937_64 mt;
    int n, m;
    uint64_t base;
    thrust::device_vector<uint64_t> dev_base_pws;
    thrust::device_vector<uint64_t> dev_s_cuts;
    std::unique_ptr<DictReader> dire;
    int q;
    thrust::device_vector<TsInfo> dev_ts_info;
    thrust::host_vector<int> hst_ts_pref_offsets;
    thrust::host_vector<int> hst_ts_count;


    E3S_sawnoff();

    void update_s(const std::vector<uint8_t>& s);

    void read_ts(std::unique_ptr<DictReader> dire);

    void compute_ts_counts();
};
