//
// Created by Radu on 8/19/2025.
//


#ifndef LOGGER_H
#define LOGGER_H

#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <limits>
#include <vector>
#include <string>
#include <bit>

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

template<typename... Args>
void log(const std::string& fmt, Args&&... args) {
    std::cerr << std::vformat(fmt, std::make_format_args(std::forward<Args>(args)...)) << '\n';
}

template<typename... Args>
void logF(const std::string& fmt, Args&&... args) {
    std::cerr << std::vformat(fmt, std::make_format_args(std::forward<Args>(args)...)) << '\n';
}

template<typename Func, typename... Args>
double fcall(const std::string& label, Func&& func, Args&&... args) {
    const auto start = std::chrono::high_resolution_clock::now();

    std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);

    const auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    auto delta = elapsed.count();
    logF("{}() done in {}ms\n", label, delta);
    return delta;
}

#endif //LOGGER_H
