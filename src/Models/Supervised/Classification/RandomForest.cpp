#include <set>
#include <map>
#include <random>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Stats/stats_reg.hpp"
#include "Stats/stats_class.hpp"
#include "Models/Supervised/Classification/RandomForest.hpp"

using namespace Utils;

namespace Class {

size_t RandomForest::max_features(size_t p) {
    if (max_features_ == "sqrt") return static_cast<size_t>(std::round(std::sqrt(p)));
    else if (max_features_ == "log2") return static_cast<size_t>(std::round(std::log2(p)));
    else if (max_features_ == "all") return p;
    else {
        // Trying to convert it to double
        double val;
        try { val = std::stod(max_features_); }
        catch (std::invalid_argument&) {
            throw std::invalid_argument("Max_features input not recognized: " + max_features_);
        }

        // Int or Float
        if (val == static_cast<int>(val)) return static_cast<size_t>(std::round(val));
        else return static_cast<size_t>(std::round(val * p));
    }
}

double RandomForest::majority_vote(const std::vector<double>& y) const {
    return detail::DecisionTree::leaf_value(y);
}

std::vector<size_t> RandomForest::bootstrap(int n) const {
    
    // Will select randomly n values
    std::mt19937 rng;
    std::vector<size_t> res_idx(n);
    rng.seed(std::random_device{}());
    auto dist = std::uniform_int_distribution<>(0 , n-1);
    for (size_t i = 0; i < static_cast<size_t>(n); i++) res_idx[i] = static_cast<size_t>(dist(rng));
    return res_idx;
}

Dataframe RandomForest::fit_without_stats(const Dataframe& x, const Dataframe& y) {
    
    // Tests
    basic_verif(x);
    basic_verif(y);
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }

    nb_categories(y);
    if (y.get_cols() > 1) {
        throw std::invalid_argument("RandomForest doesn't support Y One-Hot");
    }

    size_t n = x.get_rows();
    size_t p = x.get_cols();
    size_t max_p = max_features(p);

    // Getting ptrs to each row
    std::vector<std::vector<const double*>> X_rows(n);
    for (size_t i = 0; i < n; i++) {
        X_rows[i] = x.getRowPtrs(i);
    }

    // Create N trees
    std::vector<double> features_importance(p, 0.0);
    for (size_t i = 0; i < n_estimators_; i++) {

        // Getting idx for bootstrap
        std::vector<size_t> idx = bootstrap(n);

        // Creating our dataframe
        std::vector<double> X_v;
        std::vector<double> y_sample;
        X_v.reserve(n * p);
        y_sample.reserve(n);
        for (size_t j = 0; j < n; j++) {

            size_t row_idx = idx[j];
            for (size_t k = 0; k < p; k++) {
                X_v.push_back(*X_rows[row_idx][k]);
            }
            y_sample.push_back(y.at(row_idx));
        }
        Dataframe X = {n, p, true, std::move(X_v)};

        // Getting ptrs to each col
        std::vector<std::vector<const double*>> X_cols(p);
        for (size_t j = 0; j < p; j++) {
            X_cols[j] = X.getColumnPtrs(j);
        }

        // Create our tree
        detail::DecisionTree tree(max_p, max_depth_, min_samples_split_, min_samples_leaf_, criterion_);

        // Fit 
        tree.fit(X_cols, y_sample);
        std::vector<double> features_imp = tree.get_feature_imp();
        for (size_t j = 0; j < p; j++) features_importance[j] += features_imp[j];

        // Add it to Forest
        forest.push_back(std::move(tree));
    }

    // Results
    is_fitted = true;
    Dataframe features_imp = {1, p, false, std::move(features_importance)}; 
    return features_imp;
}

std::vector<double> RandomForest::predict(const Dataframe& x) const {
    basic_verif(x);
    if (!is_fitted) {
        throw std::runtime_error("Need to have trained your model");
    }
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }
    
    size_t n = x.get_rows();

    // Getting ptrs to each row
    std::vector<std::vector<const double*>> X_rows(n);
    for (size_t i = 0; i < n; i++) {
        X_rows[i] = x.getRowPtrs(i);
    }

    // Predict on each tree
    size_t nb_tree = forest.size();
    std::vector<std::vector<double>> forest_pred;
    forest_pred.reserve(nb_tree);
    for(size_t i = 0; i < nb_tree; i++) {
        forest_pred.push_back(
            forest[i].predict(X_rows) 
        );
    }

    // Majority vote on each obs
    std::vector<double> y_pred;
    y_pred.reserve(n);
    for (size_t i = 0; i < n; i++) {

        std::vector<double> pred_obs(nb_tree, 0.0);
        for (size_t j = 0; j < nb_tree; j++) {
            pred_obs[j] = forest_pred[j][i];
        }
        y_pred.push_back(
            majority_vote(pred_obs)
        );
    }
    return y_pred;
}

std::unique_ptr<ClassificationBase> RandomForest::create(const std::vector<std::variant<double, std::string>>& params) {

    if (params.size() == 6) {
        return std::make_unique<RandomForest>(
            std::get<double>(params[0]), 
            std::get<double>(params[1]),
            std::get<double>(params[2]), 
            std::get<double>(params[3]),
            std::get<std::string>(params[4]),
            std::get<std::string>(params[5])
        );
    }
    else throw std::invalid_argument("For RandomForest fill all inputs");
}

void RandomForest::compute_stats(const Dataframe& x, Dataframe& features_imp, const Dataframe& y) {
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    
    // Predict 
    std::vector<double> y_pred = predict(x);
    Dataframe Y_pred = {n, 1, false, std::move(y_pred)};
    
    // Confusion matrix
    std::vector<double> conf_matrix = Stats_class::conf_matrix(y.get_data(), Y_pred.get_data());

    // Roc Auc 
    double roc_auc = Stats_class::roc_auc(y.get_data(), Y_pred.get_data());

    // If we have not the cols name
    std::vector<std::string> headers(p, "");
    if (x.get_headers().empty()) {
        for (size_t i = 0; i < p; i++) headers[i] = "c" + std::to_string(i);
    }
    else {
        headers = {};
        headers.insert(headers.end(), x.get_headers().begin(), x.get_headers().end());
    }

    // Calculating features importance
    double sum = 0.0;
    std::vector<double> data = features_imp.get_data();
    for (size_t i = 0; i < data.size(); i++) sum += data[i];
    for (size_t i = 0; i < data.size(); i++) data[i] /= sum;

    // Coeff stats
    CoeffStats c;
    c.category = "";
    for (size_t i = 0; i < p; i++) {
        c.name.push_back(headers[i]);
        c.p_value.push_back(data[i]);
    }

    double TP = conf_matrix[0];
    double FN = conf_matrix[1];
    double FP = conf_matrix[2];
    double TN = conf_matrix[3];

    double prec = Stats_class::precision(TP, FP);
    double rec  = Stats_class::recall(TP, FN);
    double spec = Stats_class::specificity(TN, FP);
    double f1_  = Stats_class::f1(prec, rec);

    c.gen_stats = {prec, rec, spec, f1_, roc_auc};
    coeff_stats.push_back(c);
    gen_stats.push_back(Stats_class::mcc(conf_matrix));
    gen_stats.push_back((TP + TN) / n);
    gen_stats.push_back(prec);
    gen_stats.push_back(rec);
    gen_stats.push_back(spec);
    gen_stats.push_back(f1_);
    gen_stats.push_back(roc_auc);
}

void RandomForest::summary(bool /*detailled*/) const {

    std::cout << "\n=== Classification SUMMARY ===\n\n";

    // Global Stats
    std::cout << "MCC         = " << gen_stats[0] << "\n";
    std::cout << "Accuracy    = " << gen_stats[1] << "\n";
    std::cout << "Precision   = " << gen_stats[2] << "\n";
    std::cout << "Recall      = " << gen_stats[3] << "\n";
    std::cout << "Specificity = " << gen_stats[4] << "\n";
    std::cout << "F1          = " << gen_stats[5] << "\n";
    std::cout << "ROC AUC     = " << gen_stats[6] << "\n\n";

    std::cout << "-------  Feature Importance by IG  -------\n";

    // Coefficients table
    std::cout << std::left  << std::setw(25) << "Feature"
                << std::right << std::setw(12) << "Importance Value" << "\n";
    std::cout << std::string(42, '-') << "\n\n";

    CoeffStats stat = coeff_stats[0];
    for (size_t i = 0; i < stat.name.size(); i++) {
        std::cout << std::left  << std::setw(25) << stat.name[i]
                    << std::right << std::fixed << std::setprecision(4);
        
        std::cout << std::setw(12) << stat.p_value[i] << "\n";
    }
    std::cout << "\n" << std::endl;
}

namespace detail {

    void DecisionTree::fit(const std::vector<std::vector<const double*>>& X_cols, const std::vector<double>& y) {
        size_t p = X_cols.size();
        features_importance.assign(p, 0.0);
        root = grow(X_cols, y, 0);
    }

    std::unique_ptr<Node> DecisionTree::grow(const std::vector<std::vector<const double*>>& X_cols, 
        const std::vector<double>& y, double depth) {
        
        // Test of purity of y sample
        size_t n = y.size();
        double start = y[0];
        bool is_pure = true;
        for (size_t i = 0; i < n; i++) {
            if (y[i] != start) {
                is_pure = false;
                break;
            }
        }

        // Stop conditions
        if ((depth >= max_depth_ && max_depth_ != -1) || n < min_samples_split_ || is_pure) {
            Node node;
            node.value = leaf_value(y);
            return std::make_unique<Node>(std::move(node));
        }
        else {
            auto [best_feature, best_threshold, best_IG] = best_split(X_cols, y);
            if (isnan(best_IG)) {
                Node node;
                node.value = leaf_value(y);
                return std::make_unique<Node>(std::move(node));   
            }

            std::vector<bool> left_idx = split(X_cols[best_feature], best_threshold);

            // Create our left and right vectors
            std::vector<double> left_y;
            std::vector<double> right_y;
            for (size_t i = 0; i < n; i++) {

                if (left_idx[i]) left_y.push_back(y[i]);
                else right_y.push_back(y[i]);
            }

            // Check if left or right aren't empty or < min_samples_split_
            size_t n_left = left_y.size();
            if (n_left == n || n_left < min_samples_leaf_ || (n - n_left) < min_samples_leaf_) {
                Node node;
                node.value = leaf_value(y);
                return std::make_unique<Node>(std::move(node));
            }

            // Create our X_cols_left and right
            size_t p = X_cols.size();
            std::vector<std::vector<const double*>> left_X;
            std::vector<std::vector<const double*>> right_X;
            left_X.reserve(p);
            right_X.reserve(p);
            for (size_t i = 0; i < p; i++) {

                std::vector<const double*> left_inter;
                std::vector<const double*> right_inter;
                left_inter.reserve(n_left);
                right_inter.reserve(n - n_left);
                for (size_t j = 0; j < n; j++) {

                    if (left_idx[j]) left_inter.push_back(X_cols[i][j]);
                    else right_inter.push_back(X_cols[i][j]);
                }
                left_X.push_back(left_inter);
                right_X.push_back(right_inter);            
            }
            features_importance[best_feature] += best_IG;

            Node node;
            node.threshold = best_threshold;
            node.feature_index = best_feature;
            node.left = grow(left_X, left_y, depth + 1);
            node.right = grow(right_X, right_y, depth + 1);
            return std::make_unique<Node>(std::move(node));
        }
    }     

    double DecisionTree::leaf_value(const std::vector<double>& y) {
        
        // Getting nb of val of each category
        std::unordered_map<int, int> counts;
        for (double val : y) counts[static_cast<int>(val)]++;
        
        // Getting majority
        int majority = counts.begin()->first;
        int max_count = counts.begin()->second;
        for (auto& [label, count] : counts) {

            if (count > max_count) {
                max_count = count;
                majority = label;
            }
        }
        return majority;
    }

    std::vector<bool> DecisionTree::split(const std::vector<const double*> X_col, double threshold) const {

        size_t n = X_col.size();
        std::vector<bool> indices_left(n, false);
        for (size_t i = 0; i < n; i++) {
            if (*X_col[i] <= threshold) indices_left[i] = true;
        }
        return indices_left;
    }

    double DecisionTree::information_gain(const std::vector<double>& y, 
        const std::vector<double>& left_y, const std::vector<double>& right_y) const {

        // Calculate IG
        double stat_p;
        double stat_l;
        double stat_r;
        if (criterion_ == "entropy") {
            stat_p = entropy(y);
            stat_l = entropy(left_y);
            stat_r = entropy(right_y);
        }
        else if (criterion_ == "gini") {
            stat_p = gini(y);
            stat_l = gini(left_y);
            stat_r = gini(right_y);
        }
        else throw std::invalid_argument("Unknown criterion: " + criterion_);

        return stat_p - left_y.size() * stat_l / y.size() - right_y.size() * stat_r / y.size();
    }

    std::tuple<size_t, double, double> DecisionTree::best_split(const std::vector<std::vector<const double*>>& X_cols, 
        const std::vector<double>& y) const {
        
        // Will select randomly max_features features
        std::mt19937 rng;
        size_t n = y.size();
        size_t p = X_cols.size();
        std::vector<size_t> indices(p);
        rng.seed(std::random_device{}());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        double res_IG = -1;
        double res_threshold = -1;
        size_t res_feature = std::numeric_limits<size_t>::max();
        for (size_t i = 0; i < max_features_; i++) {
            size_t col_idx = indices[i];

            // Will get unique values in col_idx
            std::set<double> unique_val;
            for (auto x : X_cols[col_idx]) unique_val.insert(*x);
            
            // For each potential threshold split data accordingly 
            // And calculate IG to get best threshold
            double best_IG = -1;
            double best_threshold = -1;
            for (double threshold : unique_val) {

                std::vector<bool> left_idx = split(X_cols[col_idx], threshold);

                // Create our left and right vectors
                std::vector<double> left_y;
                std::vector<double> right_y;
                for (size_t j = 0; j < n; j++) {

                    if (left_idx[j]) {
                        left_y.push_back(y[j]);
                    }
                    else right_y.push_back(y[j]);
                }

                // Check if left or right aren't empty
                if (left_y.size() == n || left_y.size() == 0) continue;

                // IG
                double IG_val = information_gain(y, left_y, right_y);
                if (IG_val > best_IG) {
                    best_IG = IG_val;
                    best_threshold = threshold;
                }
            }
            if (best_IG > res_IG) {
                res_IG = best_IG;
                res_feature = col_idx;
                res_threshold = best_threshold;
            }
        }
        if (res_IG == -1) return {SIZE_MAX, NAN, NAN};
        return {res_feature, res_threshold, res_IG};
    }

    double DecisionTree::gini(const std::vector<double>& y) const {

        // Getting nb of val of each category
        std::map<int, int> counts;
        for (double x : y) {
            if (x == std::floor(x))
                counts[static_cast<int>(x)]++;
        }
        int total = static_cast<int>(y.size());

        // Getting proportions of each category
        std::map<int, double> proportions;
        for (auto& [cls, cnt] : counts)
            proportions[cls] = static_cast<double>(cnt) / total;

        // Gini
        double sum_pk = 0.0;
        for (auto& [cls, prop] : proportions) sum_pk += prop * prop; 
        return 1 - sum_pk;
    }

    double DecisionTree::entropy(const std::vector<double>& y) const {

        // Getting nb of val of each category
        std::map<int, int> counts;
        for (double x : y) {
            if (x == std::floor(x))
                counts[static_cast<int>(x)]++;
        }
        int total = static_cast<int>(y.size());

        // Getting proportions of each category
        std::map<int, double> proportions;
        for (auto& [cls, cnt] : counts)
            proportions[cls] = static_cast<double>(cnt) / total;

        // Entropy
        double sum_pk = 0.0;
        for (auto& [cls, prop] : proportions) {
            if (prop > 0.0)
                sum_pk += prop * std::log2(prop); 
        }
        return -sum_pk;
    }

    double DecisionTree::traverse(const std::vector<const double*>& X_row, Node* node) const {

        // if a node is a leaf 
        if (node->left == nullptr && node->right == nullptr) return node->value;

        // Else according to threshold go right or left
        size_t idx = node->feature_index;
        if (*X_row[idx] > node->threshold) return traverse(X_row, node->right.get());
        else return traverse(X_row, node->left.get());
    }

    std::vector<double> DecisionTree::predict(const std::vector<std::vector<const double*>>& X_rows) const {
        
        std::vector<double> y_pred;
        y_pred.reserve(X_rows.size());
        for (const auto& row : X_rows) {
            y_pred.push_back(
                traverse(row, root.get())
            );
        }
        return y_pred;
    }
}
}