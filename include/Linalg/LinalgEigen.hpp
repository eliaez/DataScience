#pragma once

#include "Data/Data.hpp"
#include <tuple>

namespace Linalg {
    namespace EigenNP {
        // Sum Eigen col - col only
        std::vector<double> sum(const std::vector<double>& v1, const std::vector<double>& v2,
            size_t m, size_t n,     // Rows / Cols
            char op = '+'           // Operator
        );

        // Mult Eigen col - col only
        std::vector<double> multiply(const std::vector<double>& v1, const std::vector<double>& v2,
            size_t m, size_t n,     // Rows / Cols v1
            size_t o, size_t p     // Rows / Cols v2
        );

        // Transpose Eigen
        Dataframe transpose(Dataframe& df);

        // Function to inverse matrix by using LU decomposition 
        Dataframe inverse(Dataframe& df);
    }
}