#pragma once

#include <cmath>
#include <vector>
#include <stdexcept>
#include <unordered_map>

#ifdef __AVX2__
    #include <immintrin.h>
#endif

namespace Utils {
    #ifdef __AVX2__
        inline constexpr size_t NB_DB = 4; // AVX2 (256 bits) so 4 doubles
        inline constexpr size_t PREFETCH_DIST = 16; // Pre-fetch 16*64 bytes ahead for contigue memory only
        inline constexpr size_t PREFETCH_DIST1 = 4; // Pre-fetch 4*64 bytes ahead for Blocks algo

        // Horizontal Reduction
        double horizontal_red(__m256d& vec);
    #endif

    // Dot product
    template<typename T>
    T dot(const std::vector<T>& x, const std::vector<T>& y);

    // Const x Vector
    template<typename T>
    std::vector<T> mult(const std::vector<T>& v, T scalar);

    // Utils
    template<typename T>
    std::vector<T> rangeExcept(T max, T exclude);

    template<typename T>
    T mostFrequent(const std::vector<T>& v);

// ----------------------------------Implementation----------------------------------

template<typename T>
T dot(const std::vector<T>& x, const std::vector<T>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Need input of same length");
    }

    T res = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        res += x[i] * y[i];
    }
    return res;
}

template<typename T>
std::vector<T> mult(const std::vector<T>& v, T scalar) {
    
    std::vector<T> result(v.size());
    for(size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}

template<typename T>
std::vector<T> rangeExcept(T max, T exclude) {
    std::vector<T> result;
    for (T i = 0; i <= max; ++i) {
        if (i != exclude) {
            result.push_back(i);
        }
    }
    return result;
}

template <typename T>
T mostFrequent(const std::vector<T>& v) {
    
    T best{};
    int maxCount = 0;
    std::unordered_map<T, int> freq;
    freq.reserve(v.size());

    for (const auto& x : v) {

        freq[x]++;
        if (freq[x] > maxCount) {
            maxCount = freq[x];
            best = x;
        }
    }
    return best;
}

}
