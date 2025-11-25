#pragma once

#include "Data/Data.hpp"


namespace Linalg {

    void sum(Dataframe&& df1, const Dataframe& df2);

    Dataframe transpose_naive(const Dataframe& df);
    Dataframe multiply_naive(Dataframe& df1, Dataframe& df2);
    
    Dataframe inverse_naive(Dataframe& df);

}; 