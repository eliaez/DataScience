#pragma once

#include "Data/Data.hpp"
#include <tuple>

namespace Linalg {
    namespace MKL {
        
        #ifdef USE_MKL
        std::vector<double> sum(const std::vector<double>& v1, const std::vector<double>& v2, // Data
            size_t m, size_t n,     // Rows / Cols
            char op = '+'           // Operator
        );

        // Mult row - col config only
        Dataframe multiply(const Dataframe& df1, const Dataframe& df2);

        // Transpose by blocks
        Dataframe transpose(Dataframe& df);

        // LU decomposition, returns nb_swaps, swap - permutation matrix and LU in the same matrix
        std::tuple<int, std::vector<double>, std::vector<double>> LU_decomposition(const Dataframe& df);

        // Function to solve LU system with Forward substitution and Backward substitution
        Dataframe solveLU_inplace(const std::vector<double>& perm, const std::vector<double>& LU, size_t n);

        // Function to inverse matrix by using LU decomposition 
        Dataframe inverse(Dataframe& df);
        #endif
    }
}