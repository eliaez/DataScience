#pragma once

#include "Data/Data.hpp"
#include <tuple>

namespace Linalg {
    namespace AVX2 {
        
        #ifdef __AVX2__
        constexpr size_t NB_DB = 4; // AVX2 (256 bits) so 4 doubles
        constexpr size_t PREFETCH_DIST = 8; // Pre-fetch 8*64 bytes ahead for contigue memory only

        Dataframe sum(const Dataframe& df1, const Dataframe& df2, char op = '+');

        // Mult AVX2 row - col config only
        Dataframe multiply(const Dataframe& df1, const Dataframe& df2);

        // Transpose by blocks
        Dataframe transpose(Dataframe& df);

        // LU decomposition, returns nb_swaps, swap - permutation matrix and LU in the same matrix
        std::tuple<int, std::vector<double>, Dataframe> LU_decomposition(const Dataframe& df);

        // Function to solve LU system with Forward substitution and Backward substitution
        Dataframe solveLU_inplace(const Dataframe& perm, const Dataframe& LU);

        // Function to inverse matrix by using LU decomposition 
        Dataframe inverse(Dataframe& df);

        // Horizontal Reduction
        double horizontal_red(__m256d& vec);
        #endif
    }
}