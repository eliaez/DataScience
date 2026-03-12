#pragma once

#include <string>

class Dataframe;

namespace Imputation {

    // Function to replace NAN 
    // by "mean" (by default)/"median"/"mode"/"forward" or "backward" Fill
    void imputation(Dataframe& x, size_t j, const std::string& method = "mean");
    void imputation(Dataframe& x, const std::string& col_name, const std::string& method = "mean");

    // Function to replace NAN with KNN (on the whole dataframe)
    // with your choice of norm L**p to calculate dist : 2 (euclidean by default)
    // If categorical colomns in dataset, it may create issues with result then use Gower distance 
    // with Lp_norm = -1
    void KNN_imputer(Dataframe& x, int K, int Lp_norm = 2);
}