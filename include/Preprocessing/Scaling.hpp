#pragma once

#include <string>

class Dataframe;

namespace Scaling {

    // Scaling function
    // Method = "standard" (z_score by default), "mean", "minmax" (0 and 1 by default), "percentile"
    void scaling(
        Dataframe& x, 
        size_t j, 
        const std::string& method = "standard",
        double min = 0,
        double max = 1
    );
    void scaling(
        Dataframe& x, 
        const std::string& col_name, 
        const std::string& method = "standard",
        double min = 0,
        double max = 1
    );

    // Function to change the general distribution of a column
    // Method = "log" (by default), "box_cox", "yeo_johnson", "power"
    void transform(
        Dataframe& x,
        size_t j,
        const std::string& method = "log",
        double lambda = 0
    );
    void transform(
        Dataframe& x,
        const std::string& col_name,
        const std::string& method = "log",
        double lambda = 0
    );
}