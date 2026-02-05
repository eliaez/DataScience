#pragma once

#include <tuple>
#include <vector>

namespace Linalg::AVX2_threaded {
    #ifdef __AVX2__

    // Sum AVX2 TH col col or row row only, will use AVX2 if threshold not crossed
    std::vector<double> sum(const std::vector<double>& v1, const std::vector<double>& v2,
        size_t m, size_t n,     // Rows / Cols
        char op = '+'           // Operator
    );

    // Mult AVX2 TH row - col config only, will use AVX2 if threshold not crossed
    std::vector<double> multiply(const std::vector<double>& v1, const std::vector<double>& v2,
        size_t m, size_t n,     // Rows / Cols v1
        size_t o, size_t p     // Rows / Cols v2
    );

    // Transpose by blocks AVX2 TH, will use AVX2 if threshold not crossed
    std::vector<double> transpose(const std::vector<double>& v1,  
        size_t v1_rows, size_t v1_cols
    );

    // LU decomposition, returns nb_swaps, swap - permutation matrix and LU in the same matrix
    std::tuple<int, std::vector<double>, std::vector<double>> LU_decomposition(const std::vector<double>& v1, size_t n);

    // Function to solve LU system with Forward substitution and Backward substitution
    std::vector<double> solveLU_inplace(const std::vector<double>& perm, const std::vector<double>& LU, size_t n);

    // Function to inverse matrix by using LU decomposition with AVX2 TH and by blocks,
    // will use AVX2 if threshold not crossed
    std::vector<double> inverse(
        const std::vector<double>& v1, 
        size_t n,
        std::vector<double> swaps,
        std::vector<double> LU
    );
    #endif
}