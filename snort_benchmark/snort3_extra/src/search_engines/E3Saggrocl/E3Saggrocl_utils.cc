#include "E3Saggrocl_utils.h"
#include <iomanip>
#include <ostream>
#include <random>

/**
 * @tparam T Sorts an array of pairs. The first element in the pairs has to be of type int/long long/__int128.
 * @param v Pointer to an array.
 * @param l First position that is sorted (v + l).
 * @param r Last position that is sorted (v + r).
 * the sort is unstable.
 */
template<typename T>
void radixSortPairs(typename std::vector<T>::iterator v, int l, int r) {
    const int base = 256;

    std::array<std::vector<T>, 2> u;
    u[0].resize(r+1); u[1].resize(r+1);
    int cnt[base] = {0};

    int i, j, z, pin;

    auto mel = std::min_element(v+l, v+r+1)->first;
    if (mel > 0) mel = 0;

    for (i = l; i <= r; i++) {
        u[0][i].first = v[i].first - mel;
        u[0][i].second = v[i].second;
    }

    int noPasses = sizeof(v[l].first); ///4 for int, 8 for ll, 16 for __int128.
    for (i = 0, pin = 0; i < noPasses; i++, pin ^= 1) {
        std::fill(cnt, cnt + base, 0);

        for (j = l; j <= r; j++) {
            cnt[(u[pin][j].first >> (i << 3)) & 255]++;
        }

        for (j = 1; j < base; j++) {
            cnt[j] += cnt[j-1];
        }

        for (j = r; j >= l; j--) {
            z = ((u[pin][j].first >> (i << 3)) & 255);
            u[pin^1][l + (--cnt[z])] = u[pin][j];
        }
    }

    for (i = l; i <= r; i++) {
        v[i].first = u[pin][i].first + mel;
        v[i].second = u[pin][i].second;
    }
}

SharedInfo::SharedInfo() {
    profiler_start = std::chrono::steady_clock::now();

    ///current time, current clock cycle count, heap address given by the OS. https://codeforces.com/blog/entry/60442
    std::seed_seq seq {
            (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
            (uint64_t) __builtin_ia32_rdtsc(),
            (uint64_t) (uintptr_t) std::make_unique<char>().get()
    };

    std::mt19937_64 mt(seq);
    std::uniform_int_distribution<uint64_t> baseDist(257, M61 - 1);
    std::uniform_int_distribution<uint64_t> otpDist(0, ULLONG_MAX);
    base = std::make_pair(baseDist(mt), baseDist(mt)); ///uniformly and randomly choose 2 bases to use.

    while (base.second == base.first) {
        base.second = baseDist(mt);
    }

    basePow[0] = base;
    for (int j = 1; (1<<j) <= maxn; j++) {
        basePow[j].first = (__int128)basePow[j-1].first * basePow[j-1].first % M61;
        basePow[j].second = (__int128)basePow[j-1].second * basePow[j-1].second % M61;
    }

    for (int j = 0; (1<<j) <= maxn; j++) {
        logOtp[j].first = otpDist(mt);
        logOtp[j].second = otpDist(mt);
    }
}

void SharedInfo::debug_profiler() {
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::microseconds>(now - profiler_start).count() * 1e-6;

    std::cout << std::fixed << std::setprecision(3) << "PROFILER. updateText %: " << profiler_timeSpent_updateText / total_time <<
                 ", updateText(part1) %: " << profiler_timeSpent_updateText_part1 / total_time <<
                 ", massSearch %: " << profiler_timeSpent_massSearch / total_time << ", total_time(s) = " << total_time <<
                 ", avg sort size = " << (profiler_sort_times_called > 0? (double)profiler_sort_total_size / profiler_sort_times_called: 0.0) <<
                 ", max sort size = " << profiler_sort_max_size << '\n' << std::flush;
}

ExpoSizeStrSrc::ExpoSizeStrSrc(SharedInfo *sharedInfo) {
    cntUncompressedNodes = 0;
    n = 0;

    this->sharedInfo = sharedInfo;

    ///OpenCL setup:
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cout << "No platforms found.\n";
        exit(0);
    }

    default_platform = platforms[0];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    std::vector<cl::Device> devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty()) {
        std::cout << "No devices found.\n";
        exit(0);
    }

    default_device = devices[0];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    context = cl::Context({default_device});

    std::ifstream kin("/home/vlad/Documents/SublimeMerge/snort3_extra/src/search_engines/E3Saggrocl/E3Saggrocl_kernel.ocl");
    std::string kernel_code(std::istreambuf_iterator<char>(kin), (std::istreambuf_iterator<char>()));
    kin.close();

    std::cout << "kernel_code.size() = " << kernel_code.size() << '\n' << std::flush;

    sources.emplace_back(kernel_code.c_str(), kernel_code.size());

    program = cl::Program(context, sources);
    if (program.build({ default_device }) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(0);
    }

    queue = cl::CommandQueue(context, default_device);

    new_s_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(uint8_t) * maxn);
    pref_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(std::pair<uint64_t, uint64_t>) * maxn);
    spad_d = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(std::pair<uint64_t, uint64_t>) * maxn);
    hh_red_d = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(uint64_t) * maxn * max_ml2);
    b_powers_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(std::pair<uint64_t, uint64_t>) * max_ml2);
    otp_d = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(std::pair<uint64_t, uint64_t>) * max_ml2);

    queue.enqueueWriteBuffer(b_powers_d, CL_TRUE, 0, sizeof(std::pair<uint64_t, uint64_t>) * max_ml2, sharedInfo->basePow);
    queue.enqueueWriteBuffer(otp_d, CL_TRUE, 0, sizeof(std::pair<uint64_t, uint64_t>) * max_ml2, sharedInfo->logOtp);
}

/**
 * We update the text on which we search the dictionary.
 * @param newS. The new string, in uint8_t format: newS[0 .. lengthNewS - 1].
 *              It already comes in lowered (case-nonsensitive) format.
 * @param lengthNewS. The length of the new string.
 * @param connections. a sorted array [(hh1, hh2)] (i.e. the fixed dictionary. by snort's design, it is different between Mpse class instances).
                       we eventually have to find DAG links hh1 -> hh2. we use it to skip over DAG links that we never have to query.
 */
void
ExpoSizeStrSrc::updateText(
    const std::vector<uint8_t> &newS,
    int lengthNewS,
    std::vector<LinkInfo> &connections
) {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    ///OpenCL KernelFunctor setup:
    static cl::KernelFunctor<int, cl::Buffer, cl::Buffer> copy_spad_bcast(cl::Kernel(program, "copy_spad_bcast"));
    static cl::KernelFunctor<int, int, std::pair<uint64_t, uint64_t>, cl::Buffer, cl::Buffer> multi_shr_mul2x_add_spad(cl::Kernel(program, "multi_shr_mul2x_add_spad"));
    static cl::KernelFunctor<int, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> cut_reduce_hash(cl::Kernel(program, "cut_reduce_hash"));

    cntUncompressedNodes = 0;
    for (int lg = 1; lg <= lengthNewS; lg <<= 1) cntUncompressedNodes += lengthNewS + 1 - lg;

    ///TODO: scoatem astea.
    if (lengthNewS > curr_maxn) {
        curr_ml2 = 1 + int(log2(lengthNewS));

        while ((1<<strE2) < lengthNewS) {
            strE2++;
        }

        hash.resize((1 << curr_ml2) * curr_ml2);
        sortedHashes.resize((1 << curr_ml2) * curr_ml2);
        id.resize((1 << curr_ml2) * curr_ml2);
        compressedGraph.resize(lengthNewS * curr_ml2 * curr_ml2);

        compressedGraphId.resize((1 << curr_ml2) * curr_ml2);
        std::fill(compressedGraphId.begin(), compressedGraphId.end(), std::make_pair(0, 0));

        curr_maxn = lengthNewS;

        hh_red_h.resize(cntUncompressedNodes);
    }

    n = lengthNewS;

    ///compute the DAG node hashes.
    cl::NDRange global_n(n), global_nlogn(cntUncompressedNodes);

    queue.enqueueWriteBuffer(new_s_d, CL_TRUE, 0, sizeof(uint8_t) * n, newS.data());
    copy_spad_bcast(cl::EnqueueArgs(queue, global_n), n, new_s_d, pref_d).wait();

    for(int pas = 1, j = 0; pas < n; pas <<= 1, j++) {
        multi_shr_mul2x_add_spad(cl::EnqueueArgs(queue, global_n), n, pas, sharedInfo->basePow[j], pref_d, spad_d).wait();
        std::swap(pref_d, spad_d);
    }

    cut_reduce_hash(cl::EnqueueArgs(queue, global_nlogn), n, b_powers_d, otp_d, pref_d, hh_red_d).wait();
    
    queue.enqueueReadBuffer(hh_red_d, CL_TRUE, 0, sizeof(uint64_t) * cntUncompressedNodes, hh_red_h.data());
    
    ///fill hash[] and sortedHashes[] from hh_red_h[].

    cntUncompressedNodes = 0;
    for (int j = 0, treeId = 0; (1<<j) <= n; j++) {
        treeId = j * (1<<strE2);

        for (int i = 0; i + (1<<j) - 1 < n; i++, treeId++) {
            hash[treeId] = hh_red_h[cntUncompressedNodes];
            sortedHashes[cntUncompressedNodes++] = std::make_pair(hash[treeId], treeId);
        }
    }

    ///sort all the DAG hashes. afterwards, we can compress the duplicates. (!! radix sort ia 40% DIN TIMP.... std::sort ia 45%.... incerc unordered_set?)
    radixSortPairs<std::pair<uint64_t, int>>(sortedHashes.begin(), 0, cntUncompressedNodes - 1);

    sharedInfo->profiler_sort_max_size = std::max(sharedInfo->profiler_sort_max_size, cntUncompressedNodes);
    sharedInfo->profiler_sort_total_size += cntUncompressedNodes;
    sharedInfo->profiler_sort_times_called++;

    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    sharedInfo->profiler_timeSpent_updateText_part1 += std::chrono::duration_cast<std::chrono::microseconds>(end1 - start).count() * 1e-6;
    sharedInfo->profiler_counter++;
    if (sharedInfo->profiler_counter % sharedInfo->profiler_debug_every == 0) sharedInfo->debug_profiler();

    int i = 0, j, z;
    while (i < cntUncompressedNodes) {
        j = i; ///go through all indexes with the same hash as subtreeHash[i].
        z = i; ///keep in z the index with the minimum id.
        while (j < cntUncompressedNodes && sortedHashes[j].first == sortedHashes[i].first) {
            z = (sortedHashes[j].second < sortedHashes[z].second? j: z);
            j++;
        }

        ///unite all other nodes with the same hash in z.
        for (; i < j; i++) {
            id[sortedHashes[i].second] = sortedHashes[z].second;
        }
    }

    i = 0;
    int cgIndexStart = 0, cgIndexCurr = 0; ///compressedGraph indexes.
    int connIndex = 0; ///pointer for the (sorted) connections[] array.

    updBset.reset();
    while (i < cntUncompressedNodes) {
        while (connIndex < (int)connections.size() && connections[connIndex].hashes.first < sortedHashes[i].first) {
            connIndex++;
        }

        if (connIndex >= (int)connections.size() || connections[connIndex].hashes.first != sortedHashes[i].first) {
            ///there is no link in connections that begins with sortedHashes[i].first. we will never query anything related, skip.
            i++;
        } else if (i < cntUncompressedNodes - max_ml2 + 1 && sortedHashes[i+max_ml2-1].first == sortedHashes[i].first) {
            ///we have at least log2(n) DAG nodes with the same hash value. keep track of their children. when there are
            ///enough DAG nodes with the same value, we can assume that most of their children are similar. we use a bitset
            ///to mark duplicates instead of calling std::unique later.

            j = i;

            int lenI = 1 << (sortedHashes[i].second >> strE2);
            while (j < cntUncompressedNodes && sortedHashes[j].first == sortedHashes[i].first) {
                ///will iterate through the children of id = sortedHashes[j].second.
                int offsetJ = sortedHashes[j].second - (1 << strE2) * (sortedHashes[j].second >> strE2);
                for (z = 0; (1<<z) < lenI && offsetJ+lenI + (1<<z)-1 < n; z++) {
                    int idChild = (1 << strE2) * z + offsetJ+lenI;
                    if (!updBset[id[idChild]]) {
                        compressedGraph[cgIndexCurr++] = id[idChild];
                        updBset[id[idChild]] = true;
                    }
                }

                j++;
            }

            updBset.reset();
            std::sort(compressedGraph.begin() + cgIndexStart, compressedGraph.begin() + cgIndexCurr);
            compressedGraphId[id[sortedHashes[i].second]] = std::make_pair(cgIndexStart, cgIndexCurr);

            cgIndexStart = cgIndexCurr;
            i = j;
        } else {
            ///there are too little (< log2(n)) DAG nodes with the same value. post compression, there aren't enough children to warrant
            ///sorting + binary-searching through the children ids. we can check each of the DAG nodes for the child in O(1) (because
            /// we know the length of the child at which we want to look), so still O(log n) per chain progression.
            i++;
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    sharedInfo->profiler_timeSpent_updateText += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
    sharedInfo->profiler_counter++;
    if (sharedInfo->profiler_counter % sharedInfo->profiler_debug_every == 0) sharedInfo->debug_profiler();
}

/**
 * For each pair (hh1, hh2) in connections, does hh1 -> hh2 appear in the text's DAG? update connections[..].found with the answers.
 * Some queried strings may have a length that is exactly a power of two. Their chains have a length of 1 (so no links).
 * They still have entries in connections (looking like (hh1, hh1)).
 * @param connections
 */
void ExpoSizeStrSrc::massSearch(std::vector<LinkInfo> &connections) {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    int z, k, l, r;
    for (auto &conn: connections) {
        conn.found = false;

        if (conn.hashes.first == conn.hashes.second) {
            ///false link. corresponding queried string's length is exactly a power of two, so the chain's length is 1.
            ///we only check if conn.hashes is present in sortedHashes.

            z = std::lower_bound(sortedHashes.begin(), sortedHashes.begin() + cntUncompressedNodes, std::make_pair(conn.hashes.first, 0)) - sortedHashes.begin();
            conn.found = (z < cntUncompressedNodes && sortedHashes[z].first == conn.hashes.first);
        } else {
            ///check if conn.hashes.first exists.
            z = std::lower_bound(sortedHashes.begin(), sortedHashes.begin() + cntUncompressedNodes, std::make_pair(conn.hashes.first, 0)) - sortedHashes.begin();
            if (z >= cntUncompressedNodes || sortedHashes[z].first != conn.hashes.first) continue;

            ///check if conn.hashes.second exists.
            k = std::lower_bound(sortedHashes.begin(), sortedHashes.begin() + cntUncompressedNodes, std::make_pair(conn.hashes.second, 0)) - sortedHashes.begin();
            if (k >= cntUncompressedNodes || sortedHashes[k].first != conn.hashes.second) continue;

            std::tie(l, r) = compressedGraphId[id[sortedHashes[z].second]]; ///get the compressedGraph interval for conn.hashes.first.

            if (l < r) {
                ///we already collected (and sorted) the hashes of the children of anybody with the hash conn.hashes.first.
                if (!std::binary_search(compressedGraph.begin() + l, compressedGraph.begin() + r, id[sortedHashes[k].second])) {
                    continue;
                }
            } else {
                ///there are few DAG nodes with a hash value of t_hashes[i] (< log2(n)). we didn't put them in compressedGraph. find them here.
                ///per DAG node, there is also only one child with the correct length conn.endExponent.
                int onlyId = id[sortedHashes[z].second], len = 1 << (sortedHashes[z].second >> strE2), offset;
                bool found = false;

                while (z < cntUncompressedNodes && !found && id[sortedHashes[z].second] == onlyId) {
                    offset = sortedHashes[z].second - (1 << strE2) * (sortedHashes[z].second >> strE2);

                    found |= (offset+len + (1 << conn.endExponent) - 1 < n && id[(1 << strE2) * conn.endExponent + offset+len] == id[sortedHashes[k].second]);
                    z++;
                }

                if (!found) {
                    continue;
                }
            }

            conn.found = true;
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    sharedInfo->profiler_timeSpent_massSearch += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
    sharedInfo->profiler_counter++;
    if (sharedInfo->profiler_counter % sharedInfo->profiler_debug_every == 0) sharedInfo->debug_profiler();
}
