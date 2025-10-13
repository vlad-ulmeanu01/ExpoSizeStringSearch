//
// Created by Radu on 8/19/2025.
//

#ifndef FASTMAP_H
#define FASTMAP_H

#include <vector>
#include "logger.h"

static size_t next_pow2(size_t x) {
    if (x <= 1) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

template<typename KeyType>
class FastSet {

public:
    FastSet() = default;

    int steps = 0;

    FastSet(const size_t n_keys, const double capacity_factor = 4.0) {
        initialize(n_keys, capacity_factor);
    }

    void initialize(const size_t n_keys, const double capacity_factor = 4.0) {
        cap   = next_pow2(static_cast<size_t>(n_keys * capacity_factor));
        mask  = cap - 1;
        _size = 0;

        entries.resize(cap);
        for (auto &e : entries) {
            e.empty = true;
        }
    }

    bool insert(KeyType key) {
        size_t idx = key & mask;
        while (!entries[idx].empty) {
            if (entries[idx].key == key) return false;
            idx = (idx + 1) & mask;
        }
        entries[idx].key   = key;
        entries[idx].empty = false;
        ++_size;
        return true;
    }

    bool find(KeyType key) {
        size_t idx = key & mask;
        while (!entries[idx].empty) {
            if (entries[idx].key == key) return true;
            idx = (idx + 1) & mask;
        }
        return false;
    }

    size_t size() const { return _size; }


private:
    struct Entry {
        KeyType key;
        char empty;
    };

    size_t cap;
    size_t mask;
    size_t _size;
    std::vector<Entry> entries;
};


template<typename KeyType, typename ValueType>
class FastMap {

public:
    FastMap() = default;


    FastMap(const size_t n_keys, const double capacity_factor = 4.0) {
        initialize(n_keys, capacity_factor);
    }

    void initialize(const size_t n_keys, const double capacity_factor = 4.0) {
        cap   = next_pow2(static_cast<size_t>(n_keys * capacity_factor));
        mask  = cap - 1;
        _size = 0;

        entries.resize(cap);
        for (auto &e : entries) {
            e.empty = true;
        }
    }

    ValueType* insert_default_get_value(KeyType key) {
        size_t idx = key & mask;
        while (!entries[idx].empty) {
            if (entries[idx].key == key) {
                return &entries[idx].value;
            }
            idx = (idx + 1) & mask;
        }
        entries[idx].key   = key;
        entries[idx].empty = false;
        ++_size;
        return &entries[idx].value;
    }

    bool insert_default(KeyType key) {
        size_t idx = key & mask;
        while (!entries[idx].empty) {
            if (entries[idx].key == key) return false;
            idx = (idx + 1) & mask;
        }
        entries[idx].key   = key;
        entries[idx].value = ValueType();
        entries[idx].empty = false;
        ++_size;
        return true;
    }

    bool insert(KeyType key, const ValueType& value) {
        size_t idx = key & mask;
        while (!entries[idx].empty) {
            if (entries[idx].key == key) return false;
            idx = (idx + 1) & mask;
        }
        entries[idx].key   = key;
        entries[idx].value = value;
        entries[idx].empty = false;
        ++_size;
        return true;
    }

    ValueType* find(KeyType key) {
        size_t idx = key & mask;
        while (!entries[idx].empty) {
            if (entries[idx].key == key) return &entries[idx].value;
            idx = (idx + 1) & mask;
        }
        return nullptr;
    }

    ValueType getValue(KeyType key) {
        size_t idx = key & mask;
        while (!entries[idx].empty) {
            if (entries[idx].key == key) return entries[idx].value;
            idx = (idx + 1) & mask;
        }
        return ValueType{};
    }

    ValueType* update_lazy(KeyType key, char notEmpty, bool& firstTime) {

        int notEmpty_i = notEmpty;

        size_t idx = key & mask;

        while (entries[idx].empty == notEmpty) {
            if (entries[idx].key == key) return &entries[idx].value;
            idx = (idx + 1) & mask;
        }
        entries[idx].key   = key;
        entries[idx].empty = notEmpty;
        firstTime          = true;
        ++_size;
        return &entries[idx].value;
    }

    void insert_or_maximize(KeyType key, ValueType value) {
        size_t idx = key & mask;
        int count = 0;
        while (!entries[idx].empty) {
            count += 1;
            if (entries[idx].key == key) {
                entries[idx].value = std::max(entries[idx].value, value);
                return;
            }
            idx = (idx + 1) & mask;
        }
        entries[idx].key   = key;
        entries[idx].value = value;
        entries[idx].empty = false;
        ++_size;
    }

    ValueType* add_or_increment(KeyType key, ValueType value) {
        size_t idx = key & mask;
        int count = 0;
        while (!entries[idx].empty) {
            count += 1;
            if (entries[idx].key == key) {
                entries[idx].value += value;
                return &entries[idx].value;
            }
            idx = (idx + 1) & mask;
        }
        entries[idx].key   = key;
        entries[idx].value = value;
        entries[idx].empty = false;
        ++_size;
        return &entries[idx].value;
    }

    size_t size() const { return _size; }


private:
    struct Entry {
        ValueType value;
        KeyType key;
        char empty;
    };

    size_t cap;
    size_t mask;
    size_t _size;
    std::vector<Entry> entries;
};


#endif //FASTMAP_H