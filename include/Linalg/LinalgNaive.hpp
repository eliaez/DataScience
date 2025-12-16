#pragma once

#include "Data/Data.hpp"
#include <tuple>

namespace Linalg {
    namespace Naive {

        Dataframe sum(const Dataframe& df1, const Dataframe& df2, char op = '+');
        Dataframe multiply(const Dataframe& df1, const Dataframe& df2);
        Dataframe transpose(Dataframe& df);

        // LU decomposition, returns nb_swaps, swap - permutation matrix and LU in the same matrix
        std::tuple<int, std::vector<double>, Dataframe> LU_decomposition(const Dataframe& df);

        // Function to solve LU system with Forward substitution and Backward substitution
        Dataframe solveLU_inplace(const Dataframe& perm, const Dataframe& LU);

        // Function to inverse matrix by using LU decomposition 
        Dataframe inverse(Dataframe& df);
    }
}