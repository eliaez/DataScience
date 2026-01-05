#pragma once

#include "Data/Data.hpp"
#include <tuple>

namespace Linalg {
    namespace Eigen {

        Dataframe sum(const Dataframe& df1, const Dataframe& df2, char op = '+');

        // Mult Eigen
        Dataframe multiply(const Dataframe& df1, const Dataframe& df2);

        // Transpose Eigen
        Dataframe transpose(Dataframe& df);

        // Function to inverse matrix by using LU decomposition 
        Dataframe inverse(Dataframe& df);
    }
}