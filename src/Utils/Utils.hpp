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
    T dot(const std::vector<T>& v, const std::vector<T>& v1);

    template<typename T>
    T dot(const std::vector<const T*>& v, const std::vector<T>& v1);

    template<typename T>
    T dot(const std::vector<const T*>& v, const std::vector<const T*>& v1);

    // Const x Vector
    template<typename T>
    std::vector<T> mult(const std::vector<T>& v, T scalar);
    
    template<typename T>
    std::vector<T> mult(const std::vector<const T*>& v, T scalar);

    // Vector + Vector
    template<typename T>
    std::vector<T> add(const std::vector<T>& v, const std::vector<T>& v1);

    // Vector - Vector
    template<typename T>
    std::vector<T> sub(const std::vector<T>& v, const std::vector<T>& v1);

    // Norm L**p
    template<typename T>
    double Lnorm(const std::vector<T>& v, int p, int pow = 1);

    template<typename T>
    double Lnorm(const std::vector<const T*>& v, int p, int pow = 1);

    template<typename T>
    double Lnorm(const std::vector<T>& v, const std::vector<T>& v1, int p, int pow = 1, char op = '+');

    template<typename T>
    double Lnorm(const std::vector<const T*>& v, const std::vector<T>& v1, int p, int pow = 1, char op = '+');

    // Norm L**p NAN proof
    template<typename T>
    double Lnorm_nan(const std::vector<const T*>& v, const std::vector<const T*>& v1, int p, int pow = 1, char op = '+');

    // Utils
    template<typename T>
    std::vector<T> rangeExcept(T max, T exclude);

    template<typename T>
    T mostFrequent(const std::vector<T>& v);

    // To check if a col is categorial or not
    bool allIntegers(const std::vector<const double*>& col);

// ----------------------------------Implementation----------------------------------

template<typename T>
T dot(const std::vector<T>& v, const std::vector<T>& v1) {
    if (v.size() != v1.size()) {
        throw std::invalid_argument("Need input of same length");
    }

    T res = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        res += v[i] * v1[i];
    }
    return res;
}

template<typename T>
T dot(const std::vector<const T*>& v, const std::vector<T>& v1) {
    if (v.size() != v1.size()) {
        throw std::invalid_argument("Need input of same length");
    }

    T res = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        res += (*v[i]) * v1[i];
    }
    return res;
}

template<typename T>
T dot(const std::vector<const T*>& v, const std::vector<const T*>& v1) {
    if (v.size() != v1.size()) {
        throw std::invalid_argument("Need input of same length");
    }

    T res = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        res += (*v[i]) * (*v1[i]);
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
std::vector<T> mult(const std::vector<const T*>& v, T scalar) {
    
    std::vector<T> result(v.size());
    for(size_t i = 0; i < v.size(); ++i) {
        result[i] = (*v[i]) * scalar;
    }
    return result;
}

template<typename T>
std::vector<T> add(const std::vector<T>& v, const std::vector<T>& v1) {

    if (v.size() != v1.size()) {
        throw std::invalid_argument("Need input of same length");
    }
    
    std::vector<T> result(v.size());
    for(size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] + v1[i];
    }
    return result;
}

template<typename T>
std::vector<T> sub(const std::vector<T>& v, const std::vector<T>& v1) {

    if (v.size() != v1.size()) {
        throw std::invalid_argument("Need input of same length");
    }
    
    std::vector<T> result(v.size());
    for(size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] - v1[i];
    }
    return result;
}

template<typename T>
double Lnorm(const std::vector<T>& v, int p, int pow) {

    double sum = 0.0;
    for (size_t i = 0; i < v.size(); i++) {
        sum += std::pow(std::abs(v[i]), p);
    }
    return std::pow(sum, (1.0 / p) * pow);
}

template<typename T>
double Lnorm(const std::vector<const T*>& v, int p, int pow) {

    double sum = 0.0;
    for (size_t i = 0; i < v.size(); i++) {
        sum += std::pow(std::abs((*v[i])), p);
    }
    return std::pow(sum, (1.0 / p) * pow);
}

template<typename T>
double Lnorm(const std::vector<T>& v, const std::vector<T>& v1, int p, int pow, char op) {

    double sum = 0.0;
    if (op == '-')
        for (size_t i = 0; i < v.size(); i++) {
            sum += std::pow(std::abs(v[i] - v1[i]), p);
        }
    else if (op == '+') {
        for (size_t i = 0; i < v.size(); i++) {
            sum += std::pow(std::abs(v[i] + v1[i]), p);
        }
    }
    return std::pow(sum, (1.0 / p) * pow);
}

template<typename T>
double Lnorm(const std::vector<const T*>& v, const std::vector<T>& v1, int p, int pow, char op) {

    double sum = 0.0;
    if (op == '-')
        for (size_t i = 0; i < v.size(); i++) {
            sum += std::pow(std::abs((*v[i]) - v1[i]), p);
        }
    else if (op == '+') {
        for (size_t i = 0; i < v.size(); i++) {
            sum += std::pow(std::abs((*v[i]) + v1[i]), p);
        }
    }
    return std::pow(sum, (1.0 / p) * pow);
}

template<typename T>
double Lnorm_nan(const std::vector<const T*>& v, const std::vector<const T*>& v1, int p, int pow, char op) {
    
    double sum = 0.0;
    double count = 0.0;
    if (op == '-')
        for (size_t i = 0; i < v.size(); i++) {
            if (!std::isnan(*v[i]) && !std::isnan(*v1[i])) {
                sum += std::pow(std::abs((*v[i]) - (*v1[i])), p);
                count++;
            }
        }
    else if (op == '+') {
        for (size_t i = 0; i < v.size(); i++) {
            if (!std::isnan(*v[i]) && !std::isnan(*v1[i])) {
                sum += std::pow(std::abs((*v[i]) + (*v1[i])), p);
                count++;
            }
        }
    }
    if (count == 0) return NAN;
    return std::pow(sum / count, (1.0 / p) * pow);
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
