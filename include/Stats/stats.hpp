#pragma once

#include <vector>

#ifdef __AVX2__
    #include <immintrin.h>
#endif

namespace Stats {
    
    #ifdef __AVX2__
        constexpr size_t NB_DB = 4; // AVX2 (256 bits) so 4 doubles
        constexpr size_t PREFETCH_DIST = 16; // Pre-fetch 16*64 bytes ahead for contigue memory only
        constexpr size_t PREFETCH_DIST1 = 4; // Pre-fetch 4*64 bytes ahead for Blocks algo

        // Horizontal Reduction
        double horizontal_red(__m256d& vec);
    #endif

    // Classical mean on a vector with Naive or AVX2
    double mean(const std::vector<double>& x);

    // Classical variance on a vector with Naive or AVX2
    double var(const std::vector<double>& x);

    // Classical covariance on a vector with Naive or AVX2
    double cov(const std::vector<double>& x, const std::vector<double>& y);
}