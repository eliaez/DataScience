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
        bool shuffle = true,
        bool show_progression = true
    );

    // GSres is composed of the best result with a vector of the best parameters
    // all_results to keep an history
    struct GSres {
        double best_score;
        std::vector<double> best_params;
        std::vector<std::pair<std::vector<double>, double>> all_results;
    };

    // Grid Search method by using CV
    // metric = "mse" (by default) / "mae" / "r2"
    // param_grid need to have all inputs from the constructor of your model in the same order example: 
    // {{0}, {1, 2, 3, 4, 5}, {2}}, one element in the sub vector if input is the parameter is fixed
    GSres GSearchCV(
        Reg::RegressionBase* model,
        const Dataframe& x, 
        const Dataframe& y,
        const std::vector<std::vector<double>>& param_grid,
        int k = 5,
        const std::string& metric = "mse",
        bool shuffle = true
    );

    // Random Search method by using CV
    // metric = "mse" (by default) / "mae" / "r2"
    // nb_iter corresponding to nb of iteration
    // range_grid is a grid to get the min and max of each parameters, with their type, it will be useful to choose
    // the correct distribution log (true) or uniform (false), and it needs to have all inputs from the constructor 
    // of your model in the same order example: {[{0, 10}, false], , [{1}, true], [{2, 5}}, true]}, one element 
    // in the sub vector if input is the parameter is fixed
    GSres RSearchCV(
        Reg::RegressionBase* model,
        const Dataframe& x, 
        const Dataframe& y,
        const std::vector<std::pair<std::vector<double>, bool>>& range_grid,
        int k = 5,
        const std::string& metric = "mse",
        int nb_iter = 50,
        bool shuffle = true
    );


    namespace detail {
        // Function to return all possible paths by recursivity
        std::vector<std::vector<double>> generate_recurCombi(
            std::vector<std::vector<double>>& result,
            std::vector<std::vector<double>>& current,
            const std::vector<std::vector<double>>& param_grid, 
            size_t param_index = 0
        );
    }
}