#pragma once

#include <string>
#include <vector>
#include <optional>
#include "Models/Supervised/Regression/RegBase.hpp"

class Dataframe;

namespace Validation {

    // CVres is composed of all scores for each folds, mean_score and std_score 
    struct CVres {
        std::vector<double> scores;
        double mean_score;
        double std_score;
    };

    // Cross Validation method
    // metric = "mse" (by default) / "mae" / "r2"
    CVres cross_validation(
        Reg::RegressionBase* model,
        const Dataframe& x, 
        const Dataframe& y,
        int k = 5,
        const std::string& metric = "mse",
        bool shuffle = true
    );

    // GSres is composed of the best result with a vector of the best parameters
    // and nb_iter to run for the recursiv function  
    struct GSres {
        double best_score;
        std::vector<double> best_params;
        std::pair<std::optional<int>, std::optional<int>> nb_iter;  
    };

    // Grid Search method by using CV
    // metric = "mse" (by default) / "mae" / "r2"
    // params_grid need to have all inputs from the constructor of your model in the same
    // order example: {{0}, {1, 2, 3, 4, 5}, {2}}
    // one element in the sub vector if input is the parameter is fixed
    GSres gridSearchCV(
        GSres* GridSearchCV,
        Reg::RegressionBase* model,
        const Dataframe& x, 
        const Dataframe& y,
        int k = 5,
        std::vector<std::vector<double>> params_grid,
        const std::string& metric = "mse",
        bool shuffle = true
    );

}