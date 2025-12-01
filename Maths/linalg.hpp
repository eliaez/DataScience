#pragma once

#include "Data/Data.hpp"
#include <tuple>
#include <cmath>

namespace Linalg {

    bool triangular_matrix(const Dataframe& df);

    // LU decomposition, L and U are stored in the same matrix
    std::tuple<int, Dataframe> LU_decomposition(const Dataframe& df);
    
    std::tuple<double, Dataframe> determinant_naive(Dataframe& df);
    
    Dataframe transpose_naive(Dataframe& df);
    
    Dataframe multiply_naive(const Dataframe& df1, Dataframe& df2);

    Dataframe inverse_naive(Dataframe& df);

}; 