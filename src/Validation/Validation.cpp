#include <cmath>
#include <random>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include "Data/Data.hpp"
#include "Stats/stats_reg.hpp"
#include "Validation/Validation.hpp"
#include "Preprocessing/TrainTestSplit.hpp"

namespace Validation {

CVres cross_validation(Reg::RegressionBase* model, const Dataframe& x, const Dataframe& y, 
    int k, const std::string& metric, bool shuffle) {
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    bool storage = x.get_storage();
    if (n != y.get_rows()) {
        throw std::invalid_argument("x and y must have the same size");
    }

    // Proportion of each fold
    int proportion = std::floor(n / k);

    // Vector of indices
    std::vector<size_t> indices(proportion);
    std::iota(indices.begin(), indices.end(), 0);

    // Random shuffle
    if (shuffle) {
        std::mt19937 rng;
        rng.seed(std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    CVres CV;
    CV.scores.reserve(k);
    std::vector<double> X_train(proportion * (k - 1) * p);
    std::vector<double> X_test(proportion * p);
    std::vector<double> y_train(proportion * (k - 1));    
    std::vector<double> y_test(proportion);

    // Split and fit
    for (size_t i = 0; i < k; i++) {
        
        size_t start_idx = proportion * i;
        size_t end_idx = proportion * (i + 1);
        size_t train_idx = 0, test_idx = 0;
        for (size_t j = 0; j < (proportion * k); j++) {

            // Split between train and test
            if (j >= start_idx && j < end_idx) {
                for (size_t l = 0; l < p; l++) {
                    X_test[l * proportion + test_idx] = storage ? x.at(indices[j] * p + l) : x.at(l * n + indices[j]);
                }
                y_test[j] = y.at(indices[j]);
                test_idx++;
            }
            else {
                for (size_t l = 0; l < p; l++) {
                    X_train[l * proportion + train_idx] = storage ? x.at(indices[j] * p + l) : x.at(l * n + indices[j]);
                }
                y_train[j] = y.at(indices[j]);
                train_idx++;
            }
        }

        Dataframe X_train_ = {proportion * (k - 1), p, false, std::move(X_train)};
        Dataframe X_test_ = {proportion, p, false, std::move(X_test)};
        Dataframe y_train_ = {proportion * (k - 1), 1, false, std::move(y_train)};

        // Fitting
        model->fit_without_stats(X_train_, y_train_);
        std::vector<double> y_pred = model->predict(X_test_);

        // Calculate the score
        double score;
        if (metric == "mse") {
            score = Stats::mse(y_test, y_pred);
        }
        else if (metric == "mae") {
            score = Stats::mae(y_test, y_pred);
        }
        else if (metric == "r2") {
            score = Stats::rsquared(y_test, y_pred);
        }
        else {
            throw std::invalid_argument("Unknown metric: " + metric);
        }

        // Save results
        CV.scores.push_back(score);
    }

    CV.mean_score = Stats::mean(CV.scores);
    CV.std_score = std::sqrt(Stats::var(CV.scores));
    return CV;
}

GSres gridSearchCV(GSres* GridSearchCV, Reg::RegressionBase* model, const Dataframe& x, const Dataframe& y, 
    int k, std::vector<std::vector<double>> param_grid, const std::string& metric, bool shuffle) {
    
    size_t n = param_grid.size();
    
    // Testing if we have only one input for each params
    bool keep = false;
    std::vector<double> grid_to_test(n);
    for (size_t i = 0; i < n; i++) {
        if (param_grid[i].size() > 1) {
            keep = true;
        }
        grid_to_test[i] = param_grid[i][0];
    }

    // Condition to stop the recusirve loop
    auto [current, target] = GridSearchCV->nb_iter;

    // If target has no value then calculate nb of iter to run
    if (!target.has_value()) {
        // To change
    }

    // Recursive
    if (keep) {
        for (size_t i = 0; i < n; i++) {

            size_t m = param_grid[i].size();
            std::vector<std::vector<double>> param_grid_x = param_grid;
            for (size_t j = 0; j < m; j++) {

                param_grid_x[i] = {param_grid[i][j]}; 
                gridSearchCV(GridSearchCV, model, x, y, k, param_grid_x, metric, shuffle);
            }        
        }
    }
    // If we have only one input for each params
    else {
        // Model with our parameters to test
        std::unique_ptr<Reg::RegressionBase> new_model = model->create(grid_to_test);

        // Cross Validation
        CVres CV = cross_validation(new_model.get(), x, y, k, metric, shuffle);

        // Adjust accordingly to the metric used
        if (metric == "mse" || metric == "mae") {
            if (GridSearchCV->best_params.empty()) {
                GridSearchCV->best_score = CV.mean_score;
                GridSearchCV->best_params = grid_to_test;
            }
            else {
                if (GridSearchCV->best_score > CV.mean_score) {
                    GridSearchCV->best_score = CV.mean_score;
                    GridSearchCV->best_params = grid_to_test;
                }
            }
        }
        else {
            if (GridSearchCV->best_params.empty()) {
                GridSearchCV->best_score = CV.mean_score;
                GridSearchCV->best_params = grid_to_test;
            }
            else {
                if (GridSearchCV->best_score < CV.mean_score) {
                    GridSearchCV->best_score = CV.mean_score;
                    GridSearchCV->best_params = grid_to_test;
                }
            }
        }

        // COndition to strop and return vals
    }
}


}