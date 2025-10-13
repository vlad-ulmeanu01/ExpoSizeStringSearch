//
// Created by johnthebrave on 10/7/2025.
//
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
#include <string.h>

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


class ES7 {

    /* Input */
    std::vector<std::string> input_queries;
    std::string input_document;

    /* Local data */
    std::vector<Query> queries;
    FastMap<uint64_t, Chunk> chunks;

    std::vector<uint32_t> single_character_counter;

    HashCalculator hc;
    uint32_t max_query_l;

    /* Holds all prefixes of lengths powers of two, for all queries */
    std::vector<FastMap<uint64_t, uint32_t>> prefixMap;
    uint32_t chPrefix[sigma];

public:

    ES7() : hc(P1), max_query_l(0) {}

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


    void compute_string_hashes(const std::string& s, const uint32_t chunk_size, uint64_t& hash1, uint64_t& hash2) {

        const uint32_t len  = s.length();
        const uint32_t A_sz = len - chunk_size;

        uint32_t gap = 1;
        chPrefix[s[0] - 'a'] = len - gap + 1;

        const uint32_t q_log_len = std::bit_width(chunk_size) - 1;
        hash1 = hc.roll_hash(s[0] - 'a' + 1, s[1] - 'a' + 1);

        gap <<= 1;
        prefixMap[0].insert(hash1, len - gap + 1);

        for (uint32_t i = 2, p = 2, start = 2; i <= q_log_len; i++, p <<= 1) {
            const uint64_t hash_t = hc.compute_hash(s, start, start + p);
            hash1 = hc.merge(hash1, start, hash_t);

            start += p;
            gap <<= 1;

            prefixMap[i - 1].insert_or_maximize(hash1, len - gap + 1);
        }

        if (A_sz) {
            hash2 = hc.compute_hash(s, A_sz, len);
        } else {
            hash2 = 0;
        }

    }


    void initialize_prefix_counter() {

        int prefixCounter[64] = {};

        for (auto& query : input_queries) {
            const uint32_t len        = query.size();
            const uint32_t chunk_size = get_chk_sz(len);
            const uint32_t q_log_len  = std::bit_width(chunk_size) - 1;
            prefixCounter[q_log_len]++;

            max_query_l = std::max(max_query_l, static_cast<uint32_t>(std::bit_width(chunk_size) - 1));
        }

        for (uint32_t i = max_query_l - 1; i > 0; i--) {
            prefixCounter[i] += prefixCounter[i + 1];
        }

        prefixMap.resize(max_query_l);

        for (uint32_t i = 1; i <= max_query_l; ++i) {
            prefixMap[i - 1].initialize(prefixCounter[i]);
        }

        memset(chPrefix, 0, sizeof(chPrefix));

    }

    void preprocess_queries() {

        initialize_prefix_counter();

        chunks.initialize(input_queries.size());
        single_character_counter.resize(sigma + 1);

        for (auto& query : input_queries) {
            const uint32_t len        = query.size();
            const uint32_t chunk_size = get_chk_sz(len);

            if (len == 1) {
                queries.push_back({0, nullptr, &single_character_counter[query[0] - 'a'], true});
                continue;
            }

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

    void check_match(const std::vector<uint64_t>& current_row, 
                     const uint32_t i,
                     const uint32_t chunk_len,
                     const uint32_t n) {

        // log("check_match: {}", i);

        Chunk* ch = chunks.find(current_row[i]);

        if (ch == nullptr) return;
        ch->lucky_queries++; /* increment for queries that are powers of two */

        for (auto it = ch->unique_lengths.begin(); it != ch->unique_lengths_end; ++it) {
            const uint32_t query_len    = *it;

            if (i + query_len > n) break;
            const uint32_t position     = i + query_len - chunk_len;
            const uint64_t snd_chk_hash = current_row[position];

            const auto qhash = hash_two_hashes(snd_chk_hash, query_len);
            const auto matched_query = ch->queries.find(qhash);
            if (matched_query == nullptr) continue;
            (*matched_query)++;
        }

    }

    void process_document() {
        const uint32_t n = input_document.length();
        const uint32_t logn = std::min(static_cast<uint32_t>(std::bit_width(n)) - 1, max_query_l);

        std::vector<uint64_t> lines[2];
        lines[0].resize(n); lines[1].resize(n);

        std::vector<int> skip(n);
        std::vector<int> hit(n);

        bool found_hits = false;

        int i, j;
        int first_hit_pos = -1;
        int longest_possible_hit_size;

        uint32_t current   = 1, prev           = 0;
        uint32_t chunk_len = 2, prev_chunk_len = 1;

        for (int i = n - 1; i >= 0; --i) {
            const uint32_t ch = input_document[i] - 'a';
            single_character_counter[ch]++;
            lines[0][i] = ch + 1;

            if (i == n - 1) {
                /* no larger string can start on the last position */
                if (i != 0) skip[i - 1] = -1; 
                skip[i]     = -1;
            } else if (chPrefix[ch]) {
                
                if (i != 0) skip[i - 1] = i;
                first_hit_pos = i;

                hit[i] = chPrefix[ch];
                found_hits = true;
            } else {
                if (i != 0) skip[i - 1] = skip[i];                
                hit[i] = 0;
            }
        }

        if (first_hit_pos == -1) {
            return;
        }

        for (j = 1; j <= logn; ++j) {

            const int limit = n - chunk_len;
            for (i = first_hit_pos; i <= limit; ) {

                auto left = i, right = std::min(i + hit[i], limit + 1);
                auto next_jump = limit + 1;

                for (auto k = left; k <= right; ++k) {
                    const uint64_t hash_left  = lines[prev][k];
                    const uint64_t hash_right = lines[prev][k + prev_chunk_len];
                    lines[current][k]         = hc.merge(hash_left, prev_chunk_len, hash_right);
                    
                    if (hit[k]) {
                        right = std::max(k + hit[k], right);
                        if (right >= limit) right = limit + 1;
                        next_jump = skip[k];
                    }
                }

                if (next_jump == -1) break;                
                i = std::max(next_jump, right);
            }

            uint32_t current_pos = first_hit_pos;

            first_hit_pos = -1;
            hit[current_pos] = prefixMap[j - 1].getValue(lines[current][current_pos]);


            if (hit[current_pos]) {
                check_match(lines[current], current_pos, chunk_len, n);
                first_hit_pos = current_pos;
            }

            while (true) {                            
                i = current_pos;

                while (true) {
                    i = skip[i];
                    
                    if (i == -1 || i > limit) {
                        i = -1;
                        break;
                    }

                    hit[i] = prefixMap[j - 1].getValue(lines[current][i]);
                    
                    if (hit[i]) {
                        check_match(lines[current], i, chunk_len, n);
                        break;
                    } 
                }

                skip[current_pos] = i;
                current_pos = i;       

                if (first_hit_pos == -1) 
                    first_hit_pos = i;

                if (i == -1) {
                    break;
                }
            }

            if (first_hit_pos == -1) {
                break;
            }
            prev_chunk_len = chunk_len;
            chunk_len <<=1;
            current ^= prev; prev ^= current; current ^= prev;
        }
    }

    void print_answers() const {
        for (const auto& query : queries) {
            std::cout << *query.answer << "\n";
        }
    }
};

int main () {

    ES7 solver = ES7();
    solver.parse_input();
    solver.preprocess_queries();
    solver.process_document();
    solver.print_answers();
    return 0;
}