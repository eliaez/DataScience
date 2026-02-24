#include <cmath>
#include <random>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include "Preprocessing/TrainTestSplit.hpp"

namespace Split {

TrainTestSplit train_test_split(const Dataframe& x, const Dataframe& y, int proportion) {

    size_t n = x.get_rows();
    size_t p = x.get_cols();
    if (n != y.get_rows()) {
        throw std::invalid_argument("x and y must have the same size");
    }
    if (proportion < 1 || proportion > 99) {
        throw std::invalid_argument("Proportion must be between 1 and 99 included");
    }

    // Get our nb
    size_t n_train = std::min(static_cast<int>((n * proportion) / 100), static_cast<int>(n - 1));
    size_t n_test = n - n_train;

    // Vector of indices
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Random shuffle
    std::mt19937 rng;
    rng.seed(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), rng);

    // Get the data for train
    std::vector<double> x_train(n_train * p);
    std::vector<double> y_train_v(n_train);
    if (x.get_storage()) {
        for (size_t i = 0; i < n_train; i++) {
            for (size_t j = 0; j < p; j++) {
                x_train[j * n_train + i] = x.at(indices[i] * p + j); 
            }
            y_train_v[i] = y.at(indices[i]);
        }
    }
    else {
        for (size_t j = 0; j < p; j++) {
            for (size_t i = 0; i < n_train; i++) {
                x_train[j * n_train + i] = x.at(j * n + indices[i]);

                if (j == 0) y_train_v[i] = y.at(indices[i]);
            }
        }
    }
    
    // Get the data for test
    std::vector<double> x_test(n_test * p);
    std::vector<double> y_test_v(n_test);
    if (x.get_storage()) {
        for (size_t i = n_train; i < n; i++) {
            for (size_t j = 0; j < p; j++) {
                x_test[j * n_test + i - n_train] = x.at(indices[i] * p + j); 
            }
            y_test_v[i - n_train] = y.at(indices[i]);
        }
    }
    else {
        for (size_t j = 0; j < p; j++) {
            for (size_t i = n_train; i < n; i++) {
                x_test[j * n_test + i - n_train] = x.at(j * n + indices[i]);

                if (j == 0) y_test_v[i - n_train] = y.at(indices[i]);
            }
        }
    }

    TrainTestSplit res;
    res.X_train = {n_train, p, false, std::move(x_train)};
    res.X_test = {n_test, p, false, std::move(x_test)};
    res.y_train = {n_train, 1, false, std::move(y_train_v)};
    res.y_test = {n_test, 1, false, std::move(y_test_v)};

    return res;
}

TrainTestValidSplit train_test_split(const Dataframe& x, const Dataframe& y, const std::pair<int, int> proportion) {

    size_t n = x.get_rows();
    size_t p = x.get_cols();
    if (n != y.get_rows()) {
        throw std::invalid_argument("x and y must have the same size");
    }
    if ((proportion.first < 1 || proportion.first > 99) || (proportion.second < 1 || proportion.second > 99)) {
        throw std::invalid_argument("Proportion must be between 1 and 99 included");
    }

    // Get our nb
    size_t n_train = std::min(static_cast<int>((n * proportion.first) / 100), static_cast<int>(n - 2));
    size_t n_valid = std::max(static_cast<int>((n * proportion.second) / 100), 1);
    size_t n_test = n - n_train - n_valid;

    if (n_test < 1) {
        throw std::invalid_argument("Not enough data for test set, change proportion");
    }

    // Vector of indices
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Random shuffle
    std::mt19937 rng;
    rng.seed(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), rng);

    // Get the data for train
    std::vector<double> x_train(n_train * p);
    std::vector<double> y_train_v(n_train);
    if (x.get_storage()) {
        for (size_t i = 0; i < n_train; i++) {
            for (size_t j = 0; j < p; j++) {
                x_train[j * n_train + i] = x.at(indices[i] * p + j); 
            }
            y_train_v[i] = y.at(indices[i]);
        }
    }
    else {
        for (size_t j = 0; j < p; j++) {
            for (size_t i = 0; i < n_train; i++) {
                x_train[j * n_train + i] = x.at(j * n + indices[i]);

                if (j == 0) y_train_v[i] = y.at(indices[i]);
            }
        }
    }

    // Get the data for valid
    std::vector<double> x_valid(n_valid * p);
    std::vector<double> y_valid_v(n_valid);
    if (x.get_storage()) {
        for (size_t i = n_train; i < (n_valid + n_train); i++) {
            for (size_t j = 0; j < p; j++) {
                x_valid[j * n_valid + i - n_train] = x.at(indices[i] * p + j); 
            }
            y_valid_v[i - n_train] = y.at(indices[i]);
        }
    }
    else {
        for (size_t j = 0; j < p; j++) {
            for (size_t i = n_train; i < (n_valid + n_train); i++) {
                x_valid[j * n_valid + i - n_train] = x.at(j * n + indices[i]);

                if (j == 0) y_valid_v[i - n_train] = y.at(indices[i]);
            }
        }
    }
    
    // Get the data for test
    std::vector<double> x_test(n_test * p);
    std::vector<double> y_test_v(n_test);
    if (x.get_storage()) {
        for (size_t i = (n_valid + n_train); i < n; i++) {
            for (size_t j = 0; j < p; j++) {
                x_test[j * n_test + i - (n_valid + n_train)] = x.at(indices[i] * p + j); 
            }
            y_test_v[i - (n_valid + n_train)] = y.at(indices[i]);
        }
    }
    else {
        for (size_t j = 0; j < p; j++) {
            for (size_t i = (n_valid + n_train); i < n; i++) {
                x_test[j * n_test + i - (n_valid + n_train)] = x.at(j * n + indices[i]);

                if (j == 0) y_test_v[i - (n_valid + n_train)] = y.at(indices[i]);
            }
        }
    }

    TrainTestValidSplit res;
    res.X_train = {n_train, p, false, std::move(x_train)};
    res.X_valid = {n_valid, p, false, std::move(x_valid)};
    res.X_test = {n_test, p, false, std::move(x_test)};
    res.y_train = {n_train, 1, false, std::move(y_train_v)};
    res.y_valid = {n_valid, 1, false, std::move(y_valid_v)};
    res.y_test = {n_test, 1, false, std::move(y_test_v)};

    return res;
}

TrainTestSplit stratified_split(const Dataframe& x, const Dataframe& y, int proportion) {
    
}
}