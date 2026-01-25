#pragma once

#include "Data/Data.hpp"
#include <vector>
#include <tuple>

namespace Linalg::AVX2 {
    #ifdef __AVX2__
    constexpr size_t NB_DB = 4; // AVX2 (256 bits) so 4 doubles
    constexpr size_t PREFETCH_DIST = 16; // Pre-fetch 16*64 bytes ahead for contigue memory only
    constexpr size_t PREFETCH_DIST1 = 4; // Pre-fetch 4*64 bytes ahead for Blocks algo

    // Sum AVX2 col col or row row only
    std::vector<double> sum(const std::vector<double>& v1, const std::vector<double>& v2,
        size_t m, size_t n,     // Rows / Cols
        char op = '+'           // Operator
    );

    // Mult AVX2 row - col config only
    std::vector<double> multiply(const std::vector<double>& v1, const std::vector<double>& v2,
        size_t m, size_t n,     // Rows / Cols v1
        size_t o, size_t p     // Rows / Cols v2
    );

    // Transpose by blocks AVX2
    std::vector<double> transpose(const std::vector<double>& v1,  
        size_t v1_rows, size_t v1_cols
    );

    // LU decomposition, returns nb_swaps, swap - permutation matrix and LU in the same matrix
    std::tuple<int, std::vector<double>, std::vector<double>> LU_decomposition(const std::vector<double>& v1, size_t n);

    // Function to solve LU system with Forward substitution and Backward substitution
    std::vector<double> solveLU_inplace(const std::vector<double>& perm, const std::vector<double>& LU, size_t n);

    // Function to inverse matrix by using LU decomposition with AVX2 and by blocks 
    std::vector<double> inverse(
        const std::vector<double>& v1, 
        size_t n,
        std::vector<double> swaps,
        std::vector<double> LU
    );

    // Horizontal Reduction
    double horizontal_red(__m256d& vec);
    #endif
}