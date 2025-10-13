#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <limits>
#include <vector>
#include <string>
#include <bit>
#include <set>

#include <unordered_set>

/* optional */
#include <format>
#include <memory>
#include <ranges>

#include <chrono>
#include <functional>
#include <immintrin.h>
#include <cstdlib>
#include <cstdint>
#include <malloc.h>

#include "hash.h"
#include "fastmap.h"
#include "logger.h"

constexpr uint32_t P1 = 31;
constexpr uint32_t M1 = 1e9 + 7;
constexpr uint32_t P2 = 53;
constexpr uint32_t M2 = 1e9 + 9;

struct Chunk {

    bool initialized;
    FastMap<uint64_t, uint32_t> queries;

    uint32_t lucky_queries;
    uint32_t num_queries;

    // std::unordered_set<uint32_t> unique_lengths;
    // std::vector / std::set unique_lengths
    std::vector<uint32_t>::iterator unique_lengths_end;
    std::vector<uint32_t> unique_lengths;

    Chunk() : initialized(false), lucky_queries(0), num_queries(0) {}

    void initialize() {
        if (initialized) return;
        std::ranges::sort(unique_lengths);
        unique_lengths_end = std::ranges::unique(unique_lengths).begin();
        queries.initialize(num_queries);
        initialized = true;
    }

};

struct ShadowHit {
    Chunk* chunk;
    uint32_t position;
    uint32_t matches;
};

inline uint32_t hash_two_hashes(const uint64_t hash1, const uint64_t hash2) {
    // Cheaply combine and fold into 32 bits
    uint64_t x = (hash1 ^ (hash2 * 0x9e3779b97f4a7c15ULL));
    x ^= (x >> 32);          // xor-fold upper and lower halves
    return static_cast<uint32_t>(x);
}

struct Query {
    uint64_t snd_chk_hash;
    Chunk* chunk;
    uint32_t* answer;
    bool lucky;
};


class ES5 {

    /* Input */
    std::vector<std::string> input_queries;
    std::string input_document;

    /* Local data */
    std::vector<Query> queries;
    FastMap<uint64_t, Chunk> chunks;

    std::vector<uint32_t> single_character_counter;

    HashCalculator hc;
    uint32_t max_query_l;

public:

    ES5() : hc(P1), max_query_l(0) {}

    void parse_input() {

        std::ios::sync_with_stdio(false); std::cin.tie(nullptr); std::cout.tie(nullptr);
        std::cin >> input_document;

        int num_queries;
        std::cin >> num_queries;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        std::string query;
        while (num_queries--) {
            std::cin >> query;
            input_queries.push_back(query);
        }

        uint32_t sz = input_document.size();
        uint32_t max_powers = 1 << static_cast<uint32_t>(std::bit_width(sz) - 1);
        hc.initialize(max_powers + 1);
    }

    static uint32_t get_chk_sz(const uint32_t n) {
        return std::bit_floor(n);
    }

    static uint64_t pack_hash(const uint32_t hash_l, const uint32_t hash_r) {
        return (static_cast<uint64_t>(hash_l) << 32) | hash_r;
    }

    void compute_string_hashes(const std::string& s, const uint32_t chunk_size, uint64_t& hash1, uint64_t& hash2) const {

        const uint32_t len  = s.length();
        const uint32_t A_sz = len - chunk_size;
        const uint32_t B_sz = chunk_size - A_sz;
        const uint32_t C_sz = len - A_sz - B_sz;

        // s = /* | ----- A ----- | ------- B ------- | (end of first chunk) | ------ C ------ | */

        if (!A_sz) {
            hash1 = hc.compute_hash(s, 0, chunk_size); hash2 = 0;
            return;
        }

        const uint64_t hashA = hc.compute_hash(s, 0, A_sz);
        const uint64_t hashB = hc.compute_hash(s, A_sz, chunk_size);
        const uint64_t hashC = hc.compute_hash(s, chunk_size, len);

        hash1 = hc.merge(hashA, B_sz, hashB);
        hash2 = hc.merge(hashB, C_sz, hashC);
    }

    void preprocess_queries() {

        chunks.initialize(input_queries.size());
        single_character_counter.resize(sigma + 1);

        uint64_t ans = 0;

        for (auto& query : input_queries) {
            const uint32_t len        = query.size();
            const uint32_t chunk_size = get_chk_sz(len);

            if (len == 1) {
                queries.push_back({0, nullptr, &single_character_counter[query[0] - 'a'], true});
                continue;
            }

            max_query_l = std::max(max_query_l, static_cast<uint32_t>(std::bit_width(chunk_size) - 1));

            uint64_t chk1_hash, chk2_hash;
            compute_string_hashes(query, chunk_size, chk1_hash, chk2_hash);

            const bool lucky = (len == chunk_size);
            Chunk* ch = chunks.insert_default_get_value(chk1_hash);

            if (!lucky) {
                ch->unique_lengths.push_back(len);
                ch->num_queries++;
            }

            queries.push_back({chk2_hash, ch, nullptr, lucky});
        }

        for (auto i = 0; i < input_queries.size(); i++) {
            auto& query = queries[i];
            Chunk* ch   = query.chunk;
            if (ch == nullptr) continue;
            if (query.lucky) {
                query.answer = &ch->lucky_queries;
            } else {
                ch->initialize();
                const auto hash = hash_two_hashes(query.snd_chk_hash, input_queries[i].length());
                query.answer = ch->queries.insert_default_get_value(hash);
            }
        }
    }

    void initialize_first_row(const uint32_t n, const std::string& doc, std::vector<uint64_t>& line) {
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t ch = doc[i] - 'a';
            single_character_counter[ch]++;
            line[i] = ch + 1;
        }
    }

    void process_document() {

        const uint32_t n = input_document.length();
        const uint32_t logn = std::min(static_cast<uint32_t>(std::bit_width(n)) - 1, max_query_l);

        std::vector<uint64_t> lines[2];
        lines[0].resize(n); lines[1].resize(n);

        initialize_first_row(n, input_document, lines[0]);

        uint32_t current   = 1, prev           = 0;
        uint32_t chunk_len = 2, prev_chunk_len = 1;

        FastMap<uint64_t, ShadowHit> cached_shadow_hits(n);
        std::vector<ShadowHit*> shadow_hits(n);
        int num_hits = 0;

        uint32_t i, j;
        for (j = 1; j <= logn; ++j) {

            const int limit = n - chunk_len;

            for (i = 0; i <= limit; ++i) {
                    const uint64_t hash_left  = lines[prev][i];
                    const uint64_t hash_right = lines[prev][i + prev_chunk_len];
                    lines[current][i]         = hc.merge(hash_left, prev_chunk_len, hash_right);
            }

            int hits = 0;
            num_hits = 0;

            for (i = 0; i + chunk_len * 2 < n; ++i) {
                const uint64_t chunk_hash = lines[current][i];
                Chunk* ch = chunks.find(lines[current][i]);
                if (ch == nullptr) continue;
                ch->lucky_queries++;

                if (ch->unique_lengths.size() == 0) continue;

                const uint64_t chunk_shadow = lines[current][i + chunk_len];
                const uint64_t match_hash   = hash_two_hashes(chunk_hash, chunk_shadow);

                const char notEmpty  = current + 2;
                bool firstTime = false;
                auto* hit = cached_shadow_hits.update_lazy(match_hash, notEmpty, firstTime);

                if (firstTime) {
                    hit->chunk    = ch;
                    hit->matches  = 1;
                    hit->position = i;
                    shadow_hits[num_hits++] = hit;
                    hits++;
                } else {
                    hit->matches++;
                    hits++;
                }
            }

            for (; i <= limit; ++i) {
                Chunk* ch = chunks.find(lines[current][i]);
                if (ch == nullptr) continue;
                ch->lucky_queries++;

                for (auto it = ch->unique_lengths.begin(); it != ch->unique_lengths_end; ++it) {
                    const uint32_t query_len    = *it;

                    if (i + query_len > n) break;
                    const uint32_t position     = i + query_len - chunk_len;
                    const uint64_t snd_chk_hash = lines[current][position];

                    const auto qhash = hash_two_hashes(snd_chk_hash, query_len);
                    const auto matched_query = ch->queries.find(qhash);
                    if (matched_query == nullptr) continue;
                    (*matched_query)++;
                }
            }

            for (i = 0; i < num_hits; ++i) {
                const ShadowHit* hit = shadow_hits[i];
                Chunk* ch = hit->chunk;

                const uint32_t chunk_start   = hit->position;
                const uint32_t count_matches = hit->matches;

                for (auto it = ch->unique_lengths.begin(); it != ch->unique_lengths_end; ++it) {
                    const uint32_t query_len    = *it;
                    const uint32_t position     = chunk_start + query_len - chunk_len;
                    const uint64_t snd_chk_hash = lines[current][position];

                    const auto qhash = hash_two_hashes(snd_chk_hash, query_len);
                    const auto matched_query = ch->queries.find(qhash);
                    if (matched_query == nullptr) continue;
                    (*matched_query) += count_matches;
                }

            }

            prev_chunk_len = chunk_len;
            chunk_len <<=1;
            current ^= prev; prev ^= current; current ^= prev;
            num_hits = 0;
        }
    }

    void print_answers() const {
        for (const auto& query : queries) {
            std::cout << *query.answer << "\n";
        }
    }
};

int main () {

    ES5 solver = ES5();
    solver.parse_input();
    solver.preprocess_queries();
    solver.process_document();
    solver.print_answers();
    return 0;
}