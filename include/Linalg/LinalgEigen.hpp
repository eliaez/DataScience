#pragma once

#include "Data/Data.hpp"
#include <tuple>

namespace Linalg {
    namespace EigenNP {

        std::vector<double> sum(const std::vector<double>& v1, const std::vector<double>& v2, // Data
            size_t m, size_t n,     // Rows / Cols
            char op = '+'           // Operator
        );

        // Mult Eigen
        Dataframe multiply(const Dataframe& df1, const Dataframe& df2);

        // Transpose Eigen
        Dataframe transpose(Dataframe& df);

        // Function to inverse matrix by using LU decomposition 
        Dataframe inverse(Dataframe& df);
    }
}