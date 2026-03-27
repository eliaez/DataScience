#include <map>
#include <cmath>
#include <random>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Stats/stats_reg.hpp"
#include "Stats/stats_class.hpp"
#include "Validation/Validation.hpp"
#include "Preprocessing/TrainTestSplit.hpp"

using namespace Utils;

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
    size_t proportion = std::floor(n / k);
    size_t train_size = proportion * (k - 1);

    // Vector of indices
    std::vector<size_t> indices(proportion * k);
    std::iota(indices.begin(), indices.end(), 0);

    // Random shuffle
    if (shuffle) {
        std::mt19937 rng;
        rng.seed(std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    CVres CV;
    CV.scores.reserve(k);
    std::vector<double> X_train(train_size * p);
    std::vector<double> X_test(proportion * p);
    std::vector<double> y_train(train_size);    
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
                y_test[test_idx] = y.at(indices[j]);
                test_idx++;
            }
            else {
                for (size_t l = 0; l < p; l++) {
                    X_train[l * train_size + train_idx] = storage ? x.at(indices[j] * p + l) : x.at(l * n + indices[j]);
                }
                y_train[train_idx] = y.at(indices[j]);
                train_idx++;
            }
        }

        Dataframe X_train_ = {train_size, p, false, std::move(X_train)};
        Dataframe X_test_ = {proportion, p, false, std::move(X_test)};
        Dataframe y_train_ = {train_size, 1, false, std::move(y_train)};

        // Fitting
        model->clean_params();
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

        // For next loop
        X_train.assign(train_size * p, 0.0);
        X_test.assign(proportion * p, 0.0);
        y_train.assign(train_size, 0.0);
        y_test.assign(proportion, 0.0);

        // Show progression
        if (show_progression) {
            if (i % 1 == 0 || i == (k - 1)) {
                std::cout << "Progress: " << (i+1) << "/" << k << " (" << (100 * (i+1) / k) << "%)\n" << std::flush;
            }
        }
    }
    std::cout << std::endl;

    model->clean_params();
    CV.mean_score = Stats::mean(CV.scores);
    CV.std_score = std::sqrt(Stats::var(CV.scores));
    return CV;
}

GSres GSearchCV(Reg::RegressionBase* model, const Dataframe& x, const Dataframe& y, 
    const std::vector<std::vector<ParamValue>>& param_grid, int k, const std::string& metric, bool shuffle) {
    
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
    std::vector<std::vector<ParamValue>> all_combi;
    all_combi.reserve(total);
    std::vector<ParamValue> current;
    detail::generate_recurCombi(all_combi, current, param_grid);

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
        if (i % 5 == 0 || i == total) {
            std::cout << "Progress: " << i << "/" << total << " (" << (100 * i / total) << "%)\n" << std::flush;
        }
    }
    std::cout << std::endl;
    return res;
}

GSres RSearchCV(Reg::RegressionBase* model, const Dataframe& x,  const Dataframe& y, 
    const std::vector<std::pair<std::vector<ParamValue>, bool>>& range_grid, int k, const std::string& metric, int nb_iter, bool shuffle) {

    // Tests
    if (range_grid.empty()) {
        throw std::invalid_argument("param_grid is empty");
    }

    bool is_minimize = false;
    if (metric == "mse" || metric == "mae") is_minimize = true;

    GSres res;
    res.best_score = is_minimize ? std::numeric_limits<double>::max() : -std::numeric_limits<double>::max();
    res.all_results.reserve(nb_iter);
    size_t n = range_grid.size();

    // Random part
    std::random_device rd;
    std::mt19937 gen(rd());

    // Let's create a dist for each param
    std::vector<bool> vect_is_string(n);
    std::vector<std::uniform_real_distribution<>> vect_r_dist(n);
    std::vector<std::uniform_int_distribution<>> vect_z_dist(n);
    for (size_t i = 0; i < n; i++){

        // To test if std::string or not 
        bool is_string = std::holds_alternative<std::string>(range_grid[i].first[0]);
        vect_is_string[i] = is_string;

        // If log distribution, then create one with min and max
        if (range_grid[i].second && range_grid[i].first.size() > 1 && !is_string) {
            double min_val = std::log10(std::get<double>(range_grid[i].first[0]));
            double max_val = std::log10(std::get<double>(range_grid[i].first[1]));
            
            vect_r_dist[i].param(
                std::uniform_real_distribution<>::param_type(min_val, max_val)
            );
        }
        // Integer dist
        else if (range_grid[i].first.size() > 1 && !is_string) {
            int min_val = std::get<double>(range_grid[i].first[0]);
            int max_val = std::get<double>(range_grid[i].first[1]);

            vect_z_dist[i].param(
                std::uniform_int_distribution<>::param_type(min_val, max_val)
            );
        }
        // Integer dist for std::string
        else if (range_grid[i].first.size() > 1 && is_string) {
            vect_z_dist[i].param(
                std::uniform_int_distribution<>::param_type(0, range_grid[i].first.size() - 1)
            );
        }
    }

    // Calculating result on each one
    std::vector<ParamValue> grid_to_test(n);
    for (size_t i = 0; i < nb_iter; i++) {

        // Let's create our grid_to_test
        for (size_t j = 0; j < n; j++) {

            // If double and not fixed param
            if (range_grid[j].first.size() > 1 && !vect_is_string[j]) {
                grid_to_test[j] = range_grid[j].second ? 
                    std::pow(10, vect_r_dist[j](gen)) : 
                    static_cast<double>(vect_z_dist[j](gen));
            }
            // If string not fixed
            else if (range_grid[j].first.size() > 1 && vect_is_string[j]) {
                grid_to_test[j] = range_grid[j].first[vect_z_dist[j](gen)];
            }
            // If fixed parameter
            else {
                grid_to_test[j] = range_grid[j].first[0];
            }
        }

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
        if (i % 5 == 0 || i == (nb_iter - 1)) {
            std::cout << "Progress: " << (i+1) << "/" << nb_iter << " (" << (100 * (i+1) / nb_iter) << "%)\n" << std::flush;
        }
    }
    std::cout << std::endl;
    return res;
}

CVres cross_validation(Class::ClassificationBase* model, const Dataframe& x, const Dataframe& y, 
    int k, const std::string& metric, bool shuffle, bool stratified, bool show_progression) {
    
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
    size_t proportion = std::floor(n / k);
    size_t train_size = proportion * (k - 1);

    // Vector of indices or map of indices by classes according stratified
    std::vector<size_t> indices(proportion * k);
    std::map<double, std::vector<size_t>> class_indices;
    if (stratified) {
        for (size_t i = 0; i < n; i++) class_indices[y.at(i)].push_back(i);
    }
    else std::iota(indices.begin(), indices.end(), 0);

    // Random shuffle
    if (shuffle) {
        std::mt19937 rng;
        rng.seed(std::random_device{}());
        if (stratified) {
            for (auto& [cls, idx] : class_indices)
                std::shuffle(idx.begin(), idx.end(), rng);
        }
        else std::shuffle(indices.begin(), indices.end(), rng);
    }

    // Build k folds stratified
    if (stratified) {
        std::vector<std::vector<size_t>> folds(k);
        for (auto& [cls, idx] : class_indices) {

            size_t fold_cls_size = idx.size() / k;
            for (size_t i = 0; i < k; i++) {

                for (size_t j = i * fold_cls_size; j < (i + 1) * fold_cls_size; j++)
                    folds[i].push_back(idx[j]);
            }
        }
        indices.clear();
        indices.reserve(proportion * k);
        for (size_t i = 0; i < k; i++)
            for (size_t j : folds[i]) indices.push_back(j);
    }

    CVres CV;
    CV.scores.reserve(k);
    std::vector<double> X_train(train_size * p);
    std::vector<double> X_test(proportion * p);
    std::vector<double> y_train(train_size);    
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
                y_test[test_idx] = y.at(indices[j]);
                test_idx++;
            }
            else {
                for (size_t l = 0; l < p; l++) {
                    X_train[l * train_size + train_idx] = storage ? x.at(indices[j] * p + l) : x.at(l * n + indices[j]);
                }
                y_train[train_idx] = y.at(indices[j]);
                train_idx++;
            }
        }

        Dataframe X_train_ = {train_size, p, false, std::move(X_train)};
        Dataframe X_test_ = {proportion, p, false, std::move(X_test)};
        Dataframe y_train_ = {train_size, 1, false, std::move(y_train)};

        // Fitting
        model->clean_params();
        model->fit_without_stats(X_train_, y_train_);

        size_t n_ = y_test.size();
        size_t nb_cats = class_indices.size();
        size_t K = (nb_cats == 2) ? 1.0 : nb_cats;

        // Calculate the score
        double score;
        if (metric == "f1" || metric == "accuracy") {
            
            std::vector<double> y_pred = model->predict(X_test_);
            Dataframe Y_pred = {n_, K, false, std::move(y_pred)};

            // Confusion matrix
            std::vector<double> conf_matrix;
            if (nb_cats == 2) conf_matrix = Stats_class::conf_matrix(y_test, Y_pred.get_data());
            else conf_matrix = Stats_class::Mult::conf_matrix_mult(y_test, Y_pred);

            // Getting accuracy or f1 if binary case
            if (nb_cats == 2) {
                double TP = conf_matrix[0];
                double TN = conf_matrix[3];

                if (metric == "f1") {
                    double FN = conf_matrix[1];
                    double FP = conf_matrix[2];
                    double prec = Stats_class::precision(TP, FP);
                    double rec  = Stats_class::recall(TP, FN);
                    score = Stats_class::f1(prec, rec);
                }
                else score = (TP + TN) / n_;
            }
            // Getting accuracy or f1 if multiple classes
            else {
                if (metric == "f1") {
                    std::vector<double> f1;
                    std::vector<double> recall;
                    std::vector<double> precision;
                    f1.reserve(K);
                    recall.reserve(K);
                    precision.reserve(K);
                    for (auto cat : rangeExcept(K, K)) {

                        double TP = conf_matrix[cat * K + cat];
                        double FP = 0, FN = 0, TN = 0;
                        for (size_t j = 0; j < K; j++) {
                            if (j != cat) {
                                FP += conf_matrix[j * K + cat];  
                                FN += conf_matrix[cat * K + j];
                            }
                        }
                        TN = n_ - TP - FP - FN;
                        precision.push_back(Stats_class::precision(TP, FP));
                        recall.push_back(Stats_class::recall(TP, FN));
                        f1.push_back(Stats_class::f1(precision.back(), recall.back()));
                    }
                    score = mean(f1);
                }
                else {
                    double count = 0.0;
                    for (size_t j = 0; j < K; j++) count += conf_matrix[j * K + j];
                    score = count / n_;
                }
            }
        }
        else if (metric == "roc_auc") {

            std::vector<double> y_proba = model->predict_proba(X_test_);
            Dataframe Y_proba = {n_, nb_cats, false, std::move(y_proba)};

            // Roc Auc 
            std::vector<double> roc_auc;
            if (nb_cats == 2) {
                std::vector<double> y_proba_bin(n_);
                for (size_t j = 0; j < n_; j++) y_proba_bin[j] = Y_proba.at(K * n_ + j);
                roc_auc.push_back(Stats_class::roc_auc(y_test, y_proba_bin));
            }
            else roc_auc = Stats_class::Mult::roc_auc_mult(y_test, Y_proba);

            // Score
            score = mean(roc_auc);
        }
        else {
            throw std::invalid_argument("Unknown metric: " + metric);
        }

        // Save results
        CV.scores.push_back(score);

        // For next loop
        X_train.assign(train_size * p, 0.0);
        X_test.assign(proportion * p, 0.0);
        y_train.assign(train_size, 0.0);
        y_test.assign(proportion, 0.0);

        // Show progression
        if (show_progression) {
            if (i % 1 == 0 || i == (k - 1)) {
                std::cout << "Progress: " << (i+1) << "/" << k << " (" << (100 * (i+1) / k) << "%)\n" << std::flush;
            }
        }
    }
    std::cout << std::endl;

    model->clean_params();
    CV.mean_score = Stats::mean(CV.scores);
    CV.std_score = std::sqrt(Stats::var(CV.scores));
    return CV;
}

GSres GSearchCV(Class::ClassificationBase* model, const Dataframe& x, const Dataframe& y, 
    const std::vector<std::vector<ParamValue>>& param_grid, int k, const std::string& metric, 
    bool shuffle, bool stratified) {
    
    // Tests
    if (param_grid.empty()) {
        throw std::invalid_argument("param_grid is empty");
    }

    // Calculate nb of totals combi
    size_t total = 1;
    for (const auto& param_val : param_grid) {
        total *= param_val.size();
    }

    GSres res;
    res.best_score = -std::numeric_limits<double>::max();
    res.all_results.reserve(total);

    // Getting all possible combinations
    std::vector<std::vector<ParamValue>> all_combi;
    all_combi.reserve(total);
    std::vector<ParamValue> current;
    detail::generate_recurCombi(all_combi, current, param_grid);

    // Calculating result on each one
    int i = 0;
    for (const auto& grid_to_test : all_combi) {

        // Model with our parameters to test
        std::unique_ptr<Class::ClassificationBase> new_model = model->create(grid_to_test);

        // Cross Validation
        CVres CV = cross_validation(new_model.get(), x, y, k, metric, shuffle, stratified, false);

        // Adjust accordingly to the metric used
        if (res.best_score < CV.mean_score) {
            res.best_score = CV.mean_score;
            res.best_params = grid_to_test;
        }

        // Update history
        res.all_results.push_back(std::make_pair(grid_to_test, CV.mean_score));

        // Show progression
        i++;
        if (i % 5 == 0 || i == total) {
            std::cout << "Progress: " << i << "/" << total << " (" << (100 * i / total) << "%)\n" << std::flush;
        }
    }
    std::cout << std::endl;
    return res;
}

GSres RSearchCV(Class::ClassificationBase* model, const Dataframe& x,  const Dataframe& y, 
    const std::vector<std::pair<std::vector<ParamValue>, bool>>& range_grid, int k, const std::string& metric, 
    int nb_iter, bool shuffle, bool stratified) {

    // Tests
    if (range_grid.empty()) {
        throw std::invalid_argument("param_grid is empty");
    }

    GSres res;
    res.best_score = -std::numeric_limits<double>::max();
    res.all_results.reserve(nb_iter);
    size_t n = range_grid.size();

    // Random part
    std::random_device rd;
    std::mt19937 gen(rd());

    // Let's create a dist for each param
    std::vector<bool> vect_is_string(n);
    std::vector<std::uniform_real_distribution<>> vect_r_dist(n);
    std::vector<std::uniform_int_distribution<>> vect_z_dist(n);
    for (size_t i = 0; i < n; i++){

        // To test if std::string or not 
        bool is_string = std::holds_alternative<std::string>(range_grid[i].first[0]);
        vect_is_string[i] = is_string;

        // If log distribution, then create one with min and max
        if (range_grid[i].second && range_grid[i].first.size() > 1 && !is_string) {
            double min_val = std::log10(std::get<double>(range_grid[i].first[0]));
            double max_val = std::log10(std::get<double>(range_grid[i].first[1]));
            
            vect_r_dist[i].param(
                std::uniform_real_distribution<>::param_type(min_val, max_val)
            );
        }
        // Integer dist
        else if (range_grid[i].first.size() > 1 && !is_string) {
            int min_val = std::get<double>(range_grid[i].first[0]);
            int max_val = std::get<double>(range_grid[i].first[1]);

            vect_z_dist[i].param(
                std::uniform_int_distribution<>::param_type(min_val, max_val)
            );
        }
        // Integer dist for std::string
        else if (range_grid[i].first.size() > 1 && is_string) {
            vect_z_dist[i].param(
                std::uniform_int_distribution<>::param_type(0, range_grid[i].first.size() - 1)
            );
        }
    }

    // Calculating result on each one
    std::vector<ParamValue> grid_to_test(n);
    for (size_t i = 0; i < nb_iter; i++) {

        // Let's create our grid_to_test
        for (size_t j = 0; j < n; j++) {

            // If double and not fixed param
            if (range_grid[j].first.size() > 1 && !vect_is_string[j]) {
                grid_to_test[j] = range_grid[j].second ? 
                    std::pow(10, vect_r_dist[j](gen)) : 
                    static_cast<double>(vect_z_dist[j](gen));
            }
            // If string not fixed
            else if (range_grid[j].first.size() > 1 && vect_is_string[j]) {
                grid_to_test[j] = range_grid[j].first[vect_z_dist[j](gen)];
            }
            // If fixed parameter
            else {
                grid_to_test[j] = range_grid[j].first[0];
            }
        }

        // Model with our parameters to test
        std::unique_ptr<Class::ClassificationBase> new_model = model->create(grid_to_test);

        // Cross Validation
        CVres CV = cross_validation(new_model.get(), x, y, k, metric, shuffle, stratified, false);

        // Adjust accordingly to the metric used
        if (res.best_score < CV.mean_score) {
            res.best_score = CV.mean_score;
            res.best_params = grid_to_test;
        }

        // Update history
        res.all_results.push_back(std::make_pair(grid_to_test, CV.mean_score));

        // Show progression
        if (i % 5 == 0 || i == (nb_iter - 1)) {
            std::cout << "Progress: " << (i+1) << "/" << nb_iter << " (" << (100 * (i+1) / nb_iter) << "%)\n" << std::flush;
        }
    }
    std::cout << std::endl;
    return res;
}

GSres GSearchCV(Reg::RegressionBase* model, const Dataframe& x, const Dataframe& y, 
    const std::vector<std::vector<double>>& param_grid, int k, const std::string& metric, bool shuffle) {
    
    std::vector<std::vector<ParamValue>> converted;
    for (const auto& v : param_grid)
        converted.push_back({v.begin(), v.end()});
    
    return GSearchCV(model, x, y, converted, k, metric, shuffle);
}

GSres RSearchCV(Reg::RegressionBase* model, const Dataframe& x,  const Dataframe& y, 
    const std::vector<std::pair<std::vector<double>, bool>>& range_grid, int k, const std::string& metric, int nb_iter, bool shuffle) {

    std::vector<std::pair<std::vector<ParamValue>, bool>> converted;
    for (const auto& [v, b] : range_grid) {
        std::vector<ParamValue> converted_v(v.begin(), v.end());
        converted.push_back(std::make_pair(converted_v, b));
    }
    return RSearchCV(model, x,  y, converted, k, metric, nb_iter, shuffle);
}

GSres GSearchCV(Class::ClassificationBase* model, const Dataframe& x, const Dataframe& y, 
    const std::vector<std::vector<double>>& param_grid, int k, const std::string& metric, bool shuffle, bool stratified) {
    
    std::vector<std::vector<ParamValue>> converted;
    for (const auto& v : param_grid)
        converted.push_back({v.begin(), v.end()});
    
    return GSearchCV(model, x, y, converted, k, metric, shuffle, stratified);
}

GSres RSearchCV(Class::ClassificationBase* model, const Dataframe& x,  const Dataframe& y, 
    const std::vector<std::pair<std::vector<double>, bool>>& range_grid, int k, const std::string& metric, int nb_iter, bool shuffle, bool stratified) {

    std::vector<std::pair<std::vector<ParamValue>, bool>> converted;
    for (const auto& [v, b] : range_grid) {
        std::vector<ParamValue> converted_v(v.begin(), v.end());
        converted.push_back(std::make_pair(converted_v, b));
    }
    return RSearchCV(model, x,  y, converted, k, metric, nb_iter, shuffle, stratified);
}
}

namespace Validation::detail {
    void generate_recurCombi(std::vector<std::vector<ParamValue>>& result, std::vector<ParamValue>& current,
        const std::vector<std::vector<ParamValue>>& param_grid, size_t param_index) {

        if (param_index == param_grid.size()) {
            result.push_back(current);
            return;
        }
        
        // Recursivity, try every parameters
        for (auto value : param_grid[param_index]) {
            current.push_back(value);              // Add value
            generate_recurCombi(result, current, param_grid, param_index + 1);
            current.pop_back();                    // Try another one
        }
    }
}