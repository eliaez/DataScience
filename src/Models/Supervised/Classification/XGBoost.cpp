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
#include "Models/Supervised/Classification/XGBoost.hpp"

using namespace Utils;

namespace Class {

std::vector<double> XGBoost::softmax(const std::vector<double>& y_pred, size_t n) const {

    std::vector<double> probs(n * nb_cats);

    for (size_t i = 0; i < n; i++) {

        // Max for numerical stability
        double max_z = -std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < nb_cats; k++)
            max_z = std::max(max_z, y_pred[k * n + i]);

        // Exp
        double denom = 0.0;
        for (size_t k = 0; k < nb_cats; k++) {
            probs[k * n + i] = std::exp(y_pred[k * n + i] - max_z);
            denom += probs[k * n + i];
        }

        for (size_t k = 0; k < nb_cats; k++)
            probs[k * n + i] /= denom;
    }
    return probs;
}

// TODO //
std::vector<double> XGBoost::get_logits(const std::vector<std::vector<const double*>>& X_rows) const {
    
    size_t n = X_rows.size();
    std::vector<double> y_pred(n * nb_cats, 0.0);
    for (size_t i = 0; i < n_estimators_; i++) {
        for (size_t k = 0; k < nb_cats; k++) {

            std::vector<double> preds = forest[i * nb_cats + k].predict(X_rows);
            for (size_t j = 0; j < n; j++) {
                y_pred[k * n + j] += learning_r_ * preds[j];
            }
        }
    }
    return y_pred;
}

// TODO //
Dataframe XGBoost::fit_without_stats(const Dataframe& x, const Dataframe& y) {
    
    // Tests
    basic_verif(x);
    basic_verif(y);
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }

    nb_categories(y);
    if (y.get_cols() > 1) {
        throw std::invalid_argument("XGBoost doesn't support Y One-Hot");
    }

    size_t n = x.get_rows();
    size_t p = x.get_cols();

    // Getting ptrs to each row
    std::vector<std::vector<const double*>> X_rows(n);
    for (size_t i = 0; i < n; i++) {
        X_rows[i] = x.getRowPtrs(i);
    }

    // Getting ptrs to each col
    std::vector<std::vector<const double*>> X_cols(p);
    for (size_t j = 0; j < p; j++) {
        X_cols[j] = x.getColumnPtrs(j);
    }

    // Assign by default Forest
    forest.clear();
    for (int i = 0; i < n_estimators_ * nb_cats; i++) {
        forest.push_back(detail::DecisionTree_());
    }

    // Create N trees
    std::vector<double> y_pred(nb_cats * n, 0.0);
    for (size_t i = 0; i < n_estimators_; i++) {

        // Softmax
        std::vector<double> probs = softmax(y_pred, n);  

        // Gradient Descent + Hessian
        std::vector<double> g(nb_cats * n);
        std::vector<double> h(nb_cats * n);
        for (size_t k = 0; k < nb_cats; k++) {
            for (size_t j = 0; j < n; j++) {
                h[k * n + j] = probs[k * n + j] * (1 - probs[k * n + j]);
                g[k * n + j] = y.at(j) == k ? probs[k * n + j] - 1 : probs[k * n + j];
            }
        }

        // Create k trees
        for (size_t k = 0; k < nb_cats; k++) {
            // Create our tree
            detail::DecisionTree_ tree(gamma_, alpha_, lambda_, max_depth_, min_child_weight_);

            // Fit 
            std::vector<double> g_k(g.begin() + k*n, g.begin() + (k+1)*n);
            std::vector<double> h_k(h.begin() + k*n, h.begin() + (k+1)*n);
            tree.fit(X_cols, g_k, h_k);

            // Add it to Forest
            forest[i * nb_cats + k] = std::move(tree);
            
            // Update logits
            std::vector<double> preds = forest[i * nb_cats + k].predict(X_rows);
            for (size_t j = 0; j < n; j++) {
                y_pred[k * n + j] += learning_r_ * preds[j];
            }
        }
    }

    // Calculating features Importance
    std::vector<double> features_importance(p, 0.0);
    for (size_t i = 0; i < n_estimators_; i++) {
        for (size_t k = 0; k < nb_cats; k++) {

            std::vector<double> features_imp = forest[i * nb_cats + k].get_feature_imp();
            for (size_t j = 0; j < p; j++) features_importance[j] += features_imp[j];
        }
    }

    // Results
    is_fitted = true;
    Dataframe features_imp = {1, p, false, std::move(features_importance)}; 
    return features_imp;
}

std::vector<double> XGBoost::predict(const Dataframe& x) const {
    basic_verif(x);
    if (!is_fitted) throw std::runtime_error("Need to have trained your model");
    if (x.get_storage()) throw std::invalid_argument("Need x col-major");

    size_t n = x.get_rows();
    std::vector<std::vector<const double*>> X_rows(n);
    for (size_t i = 0; i < n; i++) X_rows[i] = x.getRowPtrs(i);

    std::vector<double> y_pred(n);
    std::vector<double> logits = get_logits(X_rows);
    std::vector<double> probs = softmax(logits, n);
    for (size_t i = 0; i < n; i++) {        

        // Getting 
        std::vector<double> probs_k(nb_cats, 0.0);
        for (size_t k = 0; k < nb_cats; k++) {
            probs_k[k] = probs[k * n + i]; 
        }
        
        // argmax
        y_pred[i] = static_cast<double>(
            std::max_element(probs_k.begin(), probs_k.end()) - probs_k.begin()
        );
    }
    return y_pred;
}

std::vector<double> XGBoost::predict_proba(const Dataframe& x) const {
    basic_verif(x);
    if (!is_fitted) throw std::runtime_error("Need to have trained your model");
    if (x.get_storage()) throw std::invalid_argument("Need x col-major");

    size_t n = x.get_rows();
    std::vector<std::vector<const double*>> X_rows(n);
    for (size_t i = 0; i < n; i++) X_rows[i] = x.getRowPtrs(i);

    std::vector<double> logits = get_logits(X_rows);
    std::vector<double> proba = softmax(logits, n);
    return proba;
}

std::unique_ptr<ClassificationBase> XGBoost::create(const std::vector<std::variant<double, std::string>>& params) {

    if (params.size() == 6) {
        return std::make_unique<XGBoost>(
            std::get<double>(params[0]), 
            std::get<double>(params[1]),
            std::get<double>(params[2]), 
            std::get<double>(params[3]),
            std::get<double>(params[4]),
            std::get<double>(params[5])
        );
    }
    else throw std::invalid_argument("For XGBoost fill all inputs");
}

void XGBoost::compute_stats(const Dataframe& x, Dataframe& features_imp, const Dataframe& y) {
    gen_stats.clear();
    coeff_stats.clear();
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    size_t K = nb_cats == 2 ? 1 : nb_cats;
    
    // Predict 
    std::vector<double> y_proba = predict_proba(x);
    Dataframe Y_proba = {n, nb_cats, false, std::move(y_proba)};

    std::vector<double> y_pred = predict(x);
    Dataframe Y_pred = {n, 1, false, std::move(y_pred)};
    
    // Confusion matrix
    std::vector<double> conf_matrix;
    if (nb_cats == 2) conf_matrix = Stats_class::conf_matrix(y.get_data(), Y_pred.get_data());
    else conf_matrix = Stats_class::Mult::conf_matrix_mult(y.get_data(), Y_pred, K);

    // Roc Auc 
    std::vector<double> roc_auc;
    if (nb_cats == 2) {
        std::vector<double> y_proba_bin(n);
        for (size_t i = 0; i < n; i++)
            y_proba_bin[i] = Y_proba.at(n + i);
        
        roc_auc.push_back(Stats_class::roc_auc(y.get_data(), y_proba_bin));
    }
    else roc_auc = Stats_class::Mult::roc_auc_mult(y.get_data(), Y_proba);

    // If we have not the cols name
    std::vector<std::string> headers(p, "");
    if (x.get_headers().empty()) {
        for (size_t i = 1; i < p; i++) headers[i] = "c" + std::to_string(i);
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

    if (nb_cats == 2) {
        // Coeff stats
        CoeffStats c;
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

        c.gen_stats = {prec, rec, spec, f1_, roc_auc[0]};
        coeff_stats.push_back(c);

        double logL = Stats_class::logLikelihood(y.get_data(), Y_proba);
        double logloss = - logL / n;
        double accuracy = (TP + TN) / n;

        gen_stats.push_back(logL);
        gen_stats.push_back(logloss);
        gen_stats.push_back(Stats_class::mcc(conf_matrix));
        gen_stats.push_back(accuracy);
        gen_stats.push_back(prec);
        gen_stats.push_back(rec);
        gen_stats.push_back(spec);
        gen_stats.push_back(f1_);
        gen_stats.push_back(roc_auc[0]);
    }
    else {
        // For each category
        std::vector<double> f1;
        std::vector<double> recall;
        std::vector<double> precision;
        std::vector<double> specificity;
        f1.reserve(K);
        recall.reserve(K);
        precision.reserve(K);
        specificity.reserve(K);
        for (auto cat : rangeExcept(K, K)) {
            
            // Save our stats
            CoeffStats c;
            c.category = "Class " + std::to_string(cat);
            
            // To match fisher matrix order
            for (size_t i = 0; i < p; i++) {
                c.name.push_back(headers[i]);
                c.p_value.push_back(data[i]);
            }

            double TP = conf_matrix[cat * K + cat];
            double FP = 0, FN = 0, TN = 0;
            for (size_t j = 0; j < K; j++) {
                if (j != cat) {
                    FP += conf_matrix[j * K + cat];  
                    FN += conf_matrix[cat * K + j];
                }
            }
            TN = n - TP - FP - FN;
            precision.push_back(Stats_class::precision(TP, FP));
            recall.push_back(Stats_class::recall(TP, FN));
            specificity.push_back(Stats_class::specificity(TN, FP));
            f1.push_back(Stats_class::f1(precision.back(), recall.back()));

            c.gen_stats.push_back(precision.back());
            c.gen_stats.push_back(recall.back());
            c.gen_stats.push_back(specificity.back());
            c.gen_stats.push_back(f1.back());
            c.gen_stats.push_back(roc_auc[cat]);
            coeff_stats.push_back(c);
        }
        double logL = Stats_class::logLikelihood(y.get_data(), Y_proba);
        double logloss = - logL / n;

        // Accuracy
        double count = 0.0;
        for (size_t i = 0; i < K; i++) count += conf_matrix[i * K + i];
        double accuracy = count / n;

        // Save general stats
        gen_stats.push_back(logL);
        gen_stats.push_back(logloss);
        gen_stats.push_back(Stats_class::Mult::mcc_mult(conf_matrix, n, K));
        gen_stats.push_back(accuracy);
        gen_stats.push_back(mean(precision));
        gen_stats.push_back(mean(recall));
        gen_stats.push_back(mean(specificity));
        gen_stats.push_back(mean(f1));
        gen_stats.push_back(mean(roc_auc));
    }
}

void XGBoost::summary(bool detailled) const {

    std::cout << "\n=== Classification SUMMARY ===\n\n";

    // Global Stats
    std::cout << "Log-Likelyhood = " << gen_stats[0] << "\n";
    std::cout << "Log-Loss       = " << gen_stats[1] << "\n";
    std::cout << "MCC            = " << gen_stats[2] << "\n";
    std::cout << "Accuracy       = " << gen_stats[3] << "\n";
    std::cout << "Precision      = " << gen_stats[4] << "\n";
    std::cout << "Recall         = " << gen_stats[5] << "\n";
    std::cout << "Specificity    = " << gen_stats[6] << "\n";
    std::cout << "F1             = " << gen_stats[7] << "\n";
    std::cout << "ROC AUC        = " << gen_stats[8] << "\n\n";

    std::cout << "-------  Feature Importance by IG  -------\n";

    // Coefficients table
    std::cout << std::left  << std::setw(25) << "Feature"
                << std::right << std::setw(12) << "Importance Value" << "\n";
    std::cout << std::string(42, '-') << "\n\n";

    CoeffStats stat = coeff_stats[0];
    std::vector<std::pair<std::string, double>> named_Impval;
    for (size_t i = 0; i < stat.name.size(); i++) {
        named_Impval.push_back({stat.name[i], stat.p_value[i]});
    }

    std::sort(named_Impval.begin(), named_Impval.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

    for (const auto& [name, imp] : named_Impval) {
        std::cout << std::left  << std::setw(25) << name
                << std::right << std::fixed << std::setprecision(4)
                << std::setw(12) << imp << "\n";
    }

    if (detailled) {
        for (const auto& stat_ : coeff_stats) {
            std::cout << "--- " << stat_.category << " ---\n";

            // Metrics
            std::cout << "Precision = " << std::fixed << std::setprecision(4) << stat_.gen_stats[0]
                    << "  Recall = "    << stat_.gen_stats[1]
                    << "  Specificity = " << stat_.gen_stats[2]
                    << "  F1 = "        << stat_.gen_stats[3]
                    << "  ROC AUC = "   << stat_.gen_stats[4] << "\n\n";
            std::cout << "\n";
        }
    }
    std::cout << "\n" << std::endl;
}

namespace detail {

    void DecisionTree_::fit(const std::vector<std::vector<const double*>>& X_cols, 
        const std::vector<double>& g, const std::vector<double>& h) {
        total_size = g.size();
        size_t p = X_cols.size();
        features_importance.assign(p, 0.0);
        root = grow(X_cols, g, h, 0);
    }

    std::unique_ptr<Node_> DecisionTree_::grow(const std::vector<std::vector<const double*>>& X_cols, 
        const std::vector<double>& g, const std::vector<double>& h, double depth) {

        // Stop conditions
        size_t n = g.size();
        if ((depth >= max_depth_ && max_depth_ != -1) ) {
            Node_ node;
            node.value = leaf_value(g, h);
            return std::make_unique<Node_>(std::move(node));
        }
        else {
            auto [best_feature, best_threshold, best_gain, best_nan_left] = best_split(X_cols, g, h);
            if (isnan(best_gain)) {
                Node_ node;
                node.value = leaf_value(g, h);
                return std::make_unique<Node_>(std::move(node));   
            }

            std::vector<bool> left_idx = split(X_cols[best_feature], best_threshold);

            // Create our left and right vectors
            std::vector<double> left_g;
            std::vector<double> right_g;
            std::vector<double> left_h;
            std::vector<double> right_h;
            for (size_t k = 0; k < n; k++) {

                if (left_idx[k]) {
                    left_g.push_back(g[k]);
                    left_h.push_back(h[k]);
                }
                else {
                    right_g.push_back(g[k]);
                    right_h.push_back(h[k]);
                }
            }

            // Check if left or right aren't empty or < min_samples_split_
            size_t n_left = left_g.size();
            if (n_left == n || n_left < min_child_weight_ || (n - n_left) < min_child_weight_) {
                Node_ node;
                node.value = leaf_value(g, h);
                return std::make_unique<Node_>(std::move(node));
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
            if (best_gain > 0) features_importance[best_feature] +=  n * best_gain / total_size;

            Node_ node;
            node.threshold = best_threshold;
            node.feature_index = best_feature;
            node.nan_goes_left = best_nan_left;
            node.left = grow(left_X, left_g, left_h, depth + 1);
            node.right = grow(right_X, right_g, right_h, depth + 1);
            return std::make_unique<Node_>(std::move(node));
        }
    }     

    double DecisionTree_::leaf_value(const std::vector<double>& g, const std::vector<double>& h) const {
        
        // Sum G and H
        double sum_g = 0.0;
        double sum_h = 0.0;
        size_t n = g.size();
        for (size_t i = 0; i < n; i++) {
            sum_g += g[i];
            sum_h += h[i];
        }

        // If L1 reg
        if (alpha_ > 0.0) {
            double sign = sum_g > 0.0 ? 1.0 : -1.0;
            sum_g = sign * std::max(std::abs(sum_g) - alpha_, 0.0);
        }
        return - sum_g / (sum_h + lambda_);
    }

    std::vector<bool> DecisionTree_::split(const std::vector<const double*> X_col, double threshold) const {

        size_t n = X_col.size();
        std::vector<bool> indices_left(n, false);
        for (size_t i = 0; i < n; i++) {
            if (*X_col[i] <= threshold) indices_left[i] = true;
        }
        return indices_left;
    }

    std::tuple<size_t, double, double, bool> DecisionTree_::best_split(const std::vector<std::vector<const double*>>& X_cols, 
        const std::vector<double>& g, const std::vector<double>& h) const {
        
        size_t n = g.size();
        size_t p = X_cols.size();
        
        // Calculate G node and H node
        double G_node = 0.0, H_node = 0.0;
        for (size_t k = 0; k < n; k++) {
            G_node += g[k];
            H_node += h[k];
        }
        
        bool res_nan_left = true;
        double res_threshold = -1;
        size_t res_feature = std::numeric_limits<size_t>::max();
        double res_gain = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < p; i++) {

            // Separate NAN and valid data
            std::vector<size_t> valid_idx, nan_idx;
            for (size_t j = 0; j < n; j++) {
                if (std::isnan(*X_cols[i][j])) nan_idx.push_back(j);
                else valid_idx.push_back(j);
            }

            // Sort only Valid ones by features
            std::sort(valid_idx.begin(), valid_idx.end(), [&](size_t a, size_t b) {
                return *X_cols[i][a] < *X_cols[i][b];
            });

            // G and H for NAN 
            double G_nan = 0.0, H_nan = 0.0;
            for (size_t k : nan_idx) {
                G_nan += g[k];
                H_nan += h[k];
            }

            // Incremental G left and H left
            double GL = 0.0; 
            double HL = 0.0;
            bool best_nan_left = true;
            double best_threshold = -1;
            size_t nv = valid_idx.size();
            double best_gain = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j + 1 < nv; j++) {

                GL += g[valid_idx[j]];
                HL += h[valid_idx[j]];
                double GR = G_node - GL;
                double HR = H_node - HL;

                // Avoid duplicates
                if (*X_cols[i][valid_idx[j]] == *X_cols[i][valid_idx[j+1]]) continue;
                
                // Add midpoint if necessary
                double threshold = (*X_cols[i][valid_idx[j]] + *X_cols[i][valid_idx[j+1]]) / 2.0;
                
                // Gain with NAN
                double gain_nan_left  = gain(GL + G_nan, HL + H_nan, GR, HR); 
                double gain_nan_right = gain(GL, HL, GR + G_nan, HR + H_nan);
                double gain_val = std::max(gain_nan_left, gain_nan_right);
                if (gain_val > best_gain) {
                    best_gain = gain_val;
                    best_threshold = threshold;
                    best_nan_left = (gain_nan_left >= gain_nan_right);
                }
            }
            if (best_gain > res_gain) {
                res_gain = best_gain;
                res_feature = i;
                res_nan_left = best_nan_left;
                res_threshold = best_threshold;
            }
        }
        if (res_gain == -std::numeric_limits<double>::infinity()) return {SIZE_MAX, NAN, NAN, true};
        return {res_feature, res_threshold, res_gain, res_nan_left};
    }

    double DecisionTree_::traverse(const std::vector<const double*>& X_row, Node_* node) const {

        // if a node is a leaf 
        if (node->left == nullptr && node->right == nullptr) return node->value;

        // Else according to threshold go right or left
        size_t idx = node->feature_index;
        if (std::isnan(*X_row[idx])) {
            if (node->nan_goes_left) return traverse(X_row, node->left.get());
            else return traverse(X_row, node->right.get());
        }
        else {
            if (*X_row[idx] > node->threshold) return traverse(X_row, node->right.get());
            else return traverse(X_row, node->left.get());
        }
    }

    double DecisionTree_::gain(double GL, double HL, double GR, double HR) const {
        
        // If L1 reg
        double new_GL = GL;
        double new_GR = GR;
        double new_HL = HL;
        double new_HR = HR;
        if (alpha_ > 0.0) {
            new_GL = std::max(std::abs(GL) - alpha_, 0.0);
            new_GR = std::max(std::abs(GR) - alpha_, 0.0);
        }

        double left   = new_GL * new_GL / (new_HL + lambda_);
        double right  = new_GR * new_GR / (new_HR + lambda_);
        double node   = (new_GL + new_GR) * (new_GL + new_GR) / (new_HL + new_HR + lambda_);
        return 0.5 * (left + right - node) - gamma_;
    }

    std::vector<double> DecisionTree_::predict(const std::vector<std::vector<const double*>>& X_rows) const {
        
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