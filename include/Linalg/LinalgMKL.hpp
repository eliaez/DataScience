#pragma once

#include "Data/Data.hpp"
#include <tuple>

namespace Linalg {
    namespace MKL {
        
        #ifdef USE_MKL
        // Sum MKL col - col only
        std::vector<double> sum(const std::vector<double>& v1, const std::vector<double>& v2,
            size_t m, size_t n,     // Rows / Cols
            char op = '+'           // Operator
        );

        // Mult MKL col - col only
        std::vector<double> multiply(const std::vector<double>& v1, const std::vector<double>& v2,
            size_t m, size_t n,     // Rows / Cols v1
            size_t o, size_t p     // Rows / Cols v2
        );

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