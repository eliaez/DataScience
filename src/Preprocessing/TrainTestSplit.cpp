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
    bool storage = x.get_storage();
    if (n != y.get_rows()) {
        throw std::invalid_argument("x and y must have the same size");
    }
    if (proportion < 1 || proportion > 99) {
        throw std::invalid_argument("Proportion must be between 1 and 99 included");
    }
    
    // Get our nb
    size_t n_train = std::min(static_cast<size_t>(std::round((n * proportion) / 100.0)), static_cast<size_t>(n - 1));
    size_t n_test = n - n_train;

    // Vector of indices
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // Random shuffle
    std::mt19937 rng;
    rng.seed(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), rng);

    // Generic Lambda Function
    auto fill = [&](
        std::vector<double>& xv, 
        std::vector<double>& yv,
        size_t offset, 
        size_t count
        ) {
        for (size_t i = 0; i < count; i++) {
            for (size_t j = 0; j < p; j++) {
                xv[j * count + i] = storage ? x.at(indices[offset + i] * p + j) : x.at(j * n + indices[offset + i]);
            }
            yv[i] = y.at(indices[offset + i]);
        }
    };

    // Get the data for train
    std::vector<double> x_train(n_train * p);
    std::vector<double> y_train_v(n_train);
    fill(x_train, y_train_v, 0, n_train);
    
    // Get the data for test
    std::vector<double> x_test(n_test * p);
    std::vector<double> y_test_v(n_test);
    fill(x_test, y_test_v, n_train, n_test);

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
    bool storage = x.get_storage();
    if (n != y.get_rows()) {
        throw std::invalid_argument("x and y must have the same size");
    }
    if ((proportion.first < 1 || proportion.first > 99) || (proportion.second < 1 || proportion.second > 99)) {
        throw std::invalid_argument("Proportion must be between 1 and 99 included");
    }
    if (proportion.first + proportion.second >= 99) {
        throw std::invalid_argument("Sum of proportions must be < 99");
    }

    // Get our nb
    size_t n_train = std::min(static_cast<size_t>(std::round((n * proportion.first) / 100.0)), static_cast<size_t>(n - 2));
    size_t n_valid = std::max(static_cast<size_t>(std::round((n * proportion.second) / 100.0)), (size_t)1);
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

    // Generic Lambda Function
    auto fill = [&](
        std::vector<double>& xv, 
        std::vector<double>& yv,
        size_t offset, 
        size_t count
        ) {
        for (size_t i = 0; i < count; i++) {
            for (size_t j = 0; j < p; j++) {
                xv[j * count + i] = storage ? x.at(indices[offset + i] * p + j) : x.at(j * n + indices[offset + i]);
            }
            yv[i] = y.at(indices[offset + i]);
        }
    };

    // Get the data for train
    std::vector<double> x_train(n_train * p);
    std::vector<double> y_train_v(n_train);
    fill(x_train, y_train_v, 0, n_train);

    // Get the data for valid
    std::vector<double> x_valid(n_valid * p);
    std::vector<double> y_valid_v(n_valid);
    fill(x_valid, y_valid_v, n_train, n_valid);
    
    // Get the data for test
    std::vector<double> x_test(n_test * p);
    std::vector<double> y_test_v(n_test);
    fill(x_test, y_test_v, n_train + n_valid, n_test);

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
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    bool storage = x.get_storage();
    if (n != y.get_rows()) {
        throw std::invalid_argument("x and y must have the same size");
    }
    if (proportion < 1 || proportion > 99) {
        throw std::invalid_argument("Proportion must be between 1 and 99 included");
    }

    // Get our nb
    size_t n_train = std::min(static_cast<size_t>(std::round((n * proportion) / 100.0)), static_cast<size_t>(n - 1));
    size_t n_test = n - n_train;

    // Map of indices for each group
    std::unordered_map<int, std::vector<size_t>> map_indices;
    for (size_t i = 0; i < n; i++) {
        map_indices[y.at(i)].push_back(i);
    }

    // Random shuffle for each group
    std::mt19937 rng;
    rng.seed(std::random_device{}());
    for (auto& [idx, indices] : map_indices) {
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    // Verif
    for (auto& [idx, indices] : map_indices) {
        if (indices.size() < 2) {
            throw std::invalid_argument("Not enough data in each group to separate in 2 sets (one group has less than 2 members)");
        }
    }

    // Generic Lambda Function
    auto fill = [&](
        std::vector<double>& xv, 
        std::vector<double>& yv,
        const std::vector<size_t>& indices, 
        size_t nb_indice
        ) {
        for (size_t i = 0; i < nb_indice; i++) {
            for (size_t j = 0; j < p; j++) {
                xv.push_back(storage ? x.at(indices[i] * p + j) : x.at(j * n + indices[i]));
            }
            yv.push_back(y.at(indices[i]));
        }
    };

    // Get the data for train
    std::vector<double> x_train;
    std::vector<double> y_train_v;
    x_train.reserve(n_train * p);
    y_train_v.reserve(n_train);

    for (auto& [idx, indices] : map_indices) {
        
        // Get proportion of indices to take in this group
        size_t nb_indice = std::min(static_cast<size_t>(std::round((indices.size() * proportion) / 100.0)), indices.size() - 1);
        fill(x_train, y_train_v, indices, nb_indice);
        
        // Erase used indices
        indices.erase(indices.begin(), indices.begin() + nb_indice);
    }
    
    // Get the data for test
    std::vector<double> x_test;
    std::vector<double> y_test_v;
    x_test.reserve(n_test * p);
    y_test_v.reserve(n_test);

    for (auto& [idx, indices] : map_indices) {
        fill(x_test, y_test_v, indices, indices.size());
    }

    TrainTestSplit res;
    res.X_train = {n_train, p, false, std::move(x_train)};
    res.X_test = {n_test, p, false, std::move(x_test)};
    res.y_train = {n_train, 1, false, std::move(y_train_v)};
    res.y_test = {n_test, 1, false, std::move(y_test_v)};

    return res;
}

TrainTestValidSplit stratified_split(const Dataframe& x, const Dataframe& y, const std::pair<int, int> proportion) {
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    bool storage = x.get_storage();
    if (n != y.get_rows()) {
        throw std::invalid_argument("x and y must have the same size");
    }
    if ((proportion.first < 1 || proportion.first > 99) || (proportion.second < 1 || proportion.second > 99)) {
        throw std::invalid_argument("Proportion must be between 1 and 99 included");
    }
    if (proportion.first + proportion.second >= 99) {
        throw std::invalid_argument("Sum of proportions must be < 99");
    }

    // Get our nb
    size_t n_train = std::min(static_cast<size_t>(std::round((n * proportion.first) / 100.0)), static_cast<size_t>(n - 2));
    size_t n_valid = std::max(static_cast<size_t>(std::round((n * proportion.second) / 100.0)), (size_t)1);
    size_t n_test = n - n_train - n_valid;

    if (n_test < 1) {
        throw std::invalid_argument("Not enough data for test set, change proportion");
    }

    // Map of indices for each group
    std::unordered_map<int, std::vector<size_t>> map_indices;
    for (size_t i = 0; i < n; i++) {
        map_indices[y.at(i)].push_back(i);
    }

    // Random shuffle for each group
    std::mt19937 rng;
    rng.seed(std::random_device{}());
    for (auto& [idx, indices] : map_indices) {
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    // Verif and save original size
    size_t i = 0;
    std::vector<size_t> original_size(map_indices.size());
    for (auto& [idx, indices] : map_indices) {
        
        size_t size = indices.size();
        if (size < 3) {
            throw std::invalid_argument("Not enough data in each group to separate in 3 sets (one group has less than 3 members)");
        }
        original_size[i] = indices.size();
        i++;
    }

    // Generic Lambda Function
    auto fill = [&](
        std::vector<double>& xv, 
        std::vector<double>& yv,
        const std::vector<size_t>& indices, 
        size_t nb_indice
        ) {
        for (size_t i = 0; i < nb_indice; i++) {
            for (size_t j = 0; j < p; j++) {
                xv.push_back(storage ? x.at(indices[i] * p + j) : x.at(j * n + indices[i]));
            }
            yv.push_back(y.at(indices[i]));
        }
    };

    // Get the data for train
    std::vector<double> x_train;
    std::vector<double> y_train_v;
    x_train.reserve(n_train * p);
    y_train_v.reserve(n_train);

    for (auto& [idx, indices] : map_indices) {
        
        // Get proportion of indices to take in this group
        size_t nb_indice = std::min(static_cast<size_t>(std::round((indices.size() * proportion.first) / 100.0)), indices.size() - 2);
        fill(x_train, y_train_v, indices, nb_indice);
        
        // Erase used indices
        indices.erase(indices.begin(), indices.begin() + nb_indice);
    }

    // Get the data for valid
    std::vector<double> x_valid;
    std::vector<double> y_valid_v;
    x_valid.reserve(n_valid * p);
    y_valid_v.reserve(n_valid);

    size_t k = 0;
    for (auto& [idx, indices] : map_indices) {
        
        // Get proportion of indices to take in this group
        size_t nb_indice = std::min(static_cast<size_t>(std::round((original_size[k] * proportion.second) / 100.0)), indices.size() - 1);
        fill(x_valid, y_valid_v, indices, nb_indice);
        
        // Erase used indices
        indices.erase(indices.begin(), indices.begin() + nb_indice);
        k++;
    }

    // Get the data for test
    std::vector<double> x_test;
    std::vector<double> y_test_v;
    x_test.reserve(n_test * p);
    y_test_v.reserve(n_test);

    for (auto& [idx, indices] : map_indices) {
        fill(x_test, y_test_v, indices, indices.size());
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
}