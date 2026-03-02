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
    int k, const std::string& metric, bool shuffle, bool show_progression) {
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    bool storage = x.get_storage();
    if (n != y.get_rows()) {
        throw std::invalid_argument("x and y must have the same size");
    }

    if (k < 2) {
        throw std::invalid_argument("k must be >= 2");
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

        // Show progression
        i++;
        if (i % 1 == 0 || i == k) {
            std::cout << "Progress: " << i << "/" << k << " (" << (100 * i / k) << "%)\r" << std::flush;
        }
    }
    std::cout << std::endl;

    CV.mean_score = Stats::mean(CV.scores);
    CV.std_score = std::sqrt(Stats::var(CV.scores));
    return CV;
}

GSres GSearchCV(Reg::RegressionBase* model, const Dataframe& x, const Dataframe& y, 
    int k, const std::vector<std::vector<double>>& param_grid, const std::string& metric, bool shuffle) {
    
    // Tests
    if (param_grid.empty()) {
        throw std::invalid_argument("param_grid is empty");
    }

    bool is_minimize = false;
    if (metric == "mse" || metric == "mae") is_minimize = true;

    // Calculate nb of totals combi
    size_t total = 1;
    for (const auto& param_val : param_grid) {
        total *= param_val.size();
    }

    GSres res;
    res.best_score = is_minimize ? std::numeric_limits<double>::max() : -std::numeric_limits<double>::max();
    res.all_results.reserve(total);

    // Getting all possible combinations
    std::vector<std::vector<double>> all_combi;
    all_combi.reserve(total);
    std::vector<std::vector<double>> current;
    all_combi = detail::generate_recurCombi(all_combi, current, param_grid);

    // Calculating result on each one
    int i = 0;
    for (const auto& grid_to_test : all_combi) {

        // Model with our parameters to test
        std::unique_ptr<Reg::RegressionBase> new_model = model->create(grid_to_test);

        // Cross Validation
        CVres CV = cross_validation(new_model.get(), x, y, k, metric, shuffle, false);

        // Adjust accordingly to the metric used
        if (is_minimize) {
            if (res.best_score > CV.mean_score) {
                res.best_score = CV.mean_score;
                res.best_params = grid_to_test;
            }
        }
        else {
            if (res.best_score < CV.mean_score) {
                res.best_score = CV.mean_score;
                res.best_params = grid_to_test;
            }
        }

        // Update history
        res.all_results.push_back(std::make_pair(grid_to_test, CV.mean_score));

        // Show progression
        i++;
        if (i % 2 == 0 || i == total) {
            std::cout << "Progress: " << i << "/" << total << " (" << (100 * i / total) << "%)\r" << std::flush;
        }
    }
    std::cout << std::endl;
    return res;
}

namespace detail {
std::vector<std::vector<double>> generate_recurCombi(
    std::vector<std::vector<double>>& result, 
    std::vector<double>& current,
    const std::vector<std::vector<double>>& param_grid,
    size_t param_index
) {

    if (param_index == param_grid.size()) {
        result.push_back(current);
        return;
    }
    
    // Recursivity, try every parameters
    for (double value : param_grid[param_index]) {
        current.push_back(value);              // Add value
        generate_recurCombi(result, current, param_grid, param_index + 1);
        current.pop_back();                    // Try another one
    }
}
}
}