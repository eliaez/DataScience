#pragma once

#include "Data/Data.hpp"
#include <tuple>

namespace Linalg {
    namespace Naive {
        
        int triangular_matrix(const Dataframe& df);

        // LU decomposition, returns nb_swaps, swap - permutation matrix and LU in the same matrix
        std::tuple<int, std::vector<double>, Dataframe> LU_decomposition(const Dataframe& df);

        std::tuple<double, std::vector<double>, Dataframe> determinant(Dataframe& df);
        
        Dataframe transpose(Dataframe& df);

        Dataframe sum(const Dataframe& df1, const Dataframe& df2, char op = '+');
        
        Dataframe multiply(const Dataframe& df1, Dataframe& df2);

        Dataframe solveLU_inplace(const Dataframe& perm, Dataframe& LU);

        Dataframe inverse(Dataframe& df);
    }
}