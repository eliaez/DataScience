#pragma once

#include "Data/Data.hpp"


namespace Linalg {

    Dataframe transpose_naive(const Dataframe& df);
        
    Dataframe multiply_naive(const Dataframe& df1, const Dataframe& df2);
    
    Dataframe inverse_naive(Dataframe& df);

}; 