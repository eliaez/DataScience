#pragma once

#include <string>

class Dataframe;

namespace Imputation {

    // Function to replace NAN 
    // by "mean" (by default)/"median"/"mode"/"forward" or "backward" Fill
    void imputation(Dataframe& x, size_t j, const std::string& method = "mean");
    void imputation(Dataframe& x, const std::string& col_name, const std::string& method = "mean");
}