//
// Created by Radu on 8/21/2025.
//

#ifndef HASH_H
#define HASH_H

#include <iostream>
#include <cstdint>
#include <limits>
#include <vector>
#include <string>
#include <bit>
#include <set>

#include <unordered_set>
#include "logger.h"

static constexpr uint64_t MOD61 = (1ULL<<61) - 1ULL;

struct Hash {
    uint64_t left, right;
    Hash(): left(0), right(0) {}
};

inline constexpr uint8_t sigma = 26;
inline uint64_t hash_cache_3[sigma][sigma][sigma];


class HashCalculator {

public:

    uint64_t base;
    std::vector<uint64_t> powers;


    HashCalculator(uint64_t base) : base(base) {}

    void initialize(const uint32_t max_size) {
        powers.resize(max_size + 1);
        powers[0] = 1;
        for (uint32_t i = 1; i <= max_size; ++i) {
            powers[i] = mod_mul(powers[i - 1], base);
        }

        for (uint8_t i = 0; i < sigma; i++) {
            for (uint8_t j = 0; j < sigma; j++) {
                for (uint8_t k = 0; k < sigma; k++) {
                    hash_cache_3[i][j][k] = roll_hash((i + 1) * base + (j + 1), (k + 1));
                }
            }
        }
    }

    uint64_t compute_hash(const std::string& str, const uint32_t start, const uint32_t end) const {

        uint64_t hash = 0;

        uint32_t i, j;
        for (i = start; i + 5 < end; i += 6) {
            const uint8_t n1 = str[i] - 'a';
            const uint8_t n2 = str[i + 1] - 'a';
            const uint8_t n3 = str[i + 2] - 'a';
            const uint8_t n4 = str[i + 3] - 'a';
            const uint8_t n5 = str[i + 4] - 'a';
            const uint8_t n6 = str[i + 5] - 'a';

            const uint64_t hash_left       = hash_cache_3[n1][n2][n3];
            const uint64_t hash_right      = hash_cache_3[n4][n5][n6];
            const uint64_t next_chunk_hash = merge(hash_left, 3, hash_right);

            hash = merge(hash, 6, next_chunk_hash);
        }

        if (i + 3 < end) {
            const uint8_t n1 = str[i] - 'a';
            const uint8_t n2 = str[i + 1] - 'a';
            const uint8_t n3 = str[i + 2] - 'a';
            hash = merge(hash, 3, hash_cache_3[n1][n2][n3]);
            i += 3;
        }

        for (j = i; j < end; j ++) {
            hash = roll_hash(hash, str[j] - 'a' + 1);
        }
        return hash;
    }

    inline uint64_t merge(const uint64_t hash_left, const uint32_t hash_right_len, const uint64_t hash_right) const {
        return mod_add(hash_right, mod_mul(hash_left, powers[hash_right_len]));
    }

    inline uint64_t roll_hash(const uint64_t hash, const unsigned char c) const {
        return mod_add(c, mod_mul(hash, base));
    }

    static inline uint64_t mod_mul(const uint64_t a, const uint64_t b) {
        __uint128_t z = static_cast<__uint128_t>(a) * b;
        // split high and low parts relative to 61 bits
        const uint64_t low     = static_cast<uint64_t>(z) & MOD61; // lower 61 bits
        const uint64_t high    = static_cast<uint64_t>(z >> 61);   // remaining high bits
        uint64_t res           = low + high;
        if (res >= MOD61) res -= MOD61;
        return res;
    }

    static inline uint64_t mod_add(uint64_t a, uint64_t b) {
        const uint64_t result = a + b;
        return (result >= MOD61) ? result - MOD61 : result;
    }

    static inline uint64_t mod_sub(uint64_t a, uint64_t b) {
        return (a >= b) ? a - b : MOD61 - (b - a);
    }

};


#endif //HASH_H