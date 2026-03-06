#include <set>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Stats/stats_reg.hpp"
#include "Validation/Validation.hpp"
#include "Models/Supervised/Regression/StepwiseReg.hpp"

using namespace Utils;

namespace Reg {

std::pair<Dataframe, Dataframe> StepwiseRegression::fit_without_stats(const Dataframe& x, const Dataframe& y) {
    
    // Tests
    basic_verif(x);
    basic_verif(y);
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }

    size_t n = x.get_rows();
    std::vector<double> x_v(n, 1.0);
    if (method_ == "backward") {
        x_v = backward_reg(x, y);
    }
    else if (method_ == "forward") {
        x_v = forward_reg(x, y);
    }
    else if (method_ == "stepwise") {
        x_v = stepwise_reg(x, y);
    }
    else {
        throw std::invalid_argument(std::format("Unknown method {}", method_));
    }

    // Need X col-major (for mult ops)
    size_t p = x_v.size() / n;
    Dataframe X = {n, p, false, std::move(x_v)};

    // Need X_t row-major (for mult ops)
    Dataframe X_t = ~X;  // Transpose change it to col-major
    X_t.change_layout_inplace();

    // Calculate Beta (our estimator)
    Dataframe XtXInv = (X_t*X).inv();
    XtXInv.change_layout_inplace();
    Dataframe beta_est =  XtXInv * (X_t * y);  

    // Getting coeffs with selected features
    clean_params();
    std::vector<double> res_coeffs(x.get_cols() + 1, 0.0);

    // Mapping
    std::unordered_map<size_t, size_t> feature_to_beta_idx;
    for (size_t i = 0; i < selected_features.size(); i++) {
        feature_to_beta_idx[selected_features[i]] = i + 1; // +1 for intercept
    }

    // Filling in order
    for (size_t i = 0; i < x.get_cols(); i++) {
        auto it = feature_to_beta_idx.find(i);
        if (it != feature_to_beta_idx.end()) {
            res_coeffs[i + 1] = beta_est.at(it->second);
        }
    }

    // Results
    res_coeffs[0] = beta_est.at(0);
    coeffs = res_coeffs;
    is_fitted = true;

    return {X, XtXInv};
}

std::vector<double> StepwiseRegression::stepwise_reg(const Dataframe& x, const Dataframe& y) {
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    std::vector<double> x_v(n, 1.0);
    std::vector<double> features_kept;

    // Getting ptrs to elem of each col for coordinate descent
    std::vector<std::vector<const double*>> X_j(p);
    for (size_t j = 0; j < p; j++) {
        X_j[j] = x.getColumnPtrs(j);
    }

    // Need to do a linear reg with just intercept to begin
    Dataframe X_try = {n, 1, false, x_v};

    // Linear Reg
    Dataframe X_t = ~X_try;  
    X_t.change_layout_inplace();
    Dataframe XtXInv = (X_t*X_try).inv();
    XtXInv.change_layout_inplace();
    Dataframe beta_est =  XtXInv * (X_t * y);

    clean_params();
    coeffs = beta_est.get_data();
    is_fitted = true;

    // Variables for threshold
    double score; // AIC or BIC
    double alpha_in;
    double alpha_out;
    double RSS_baseline = Stats::mse(y.get_data(), predict(X_try)) * n;
    if (threshold_ == "alpha" && n > 30) {
        alpha_in = alpha_.first;
        alpha_out = alpha_.second;
    }
    else if (threshold_ == "aic" || threshold_ == "bic") {
        score = (threshold_ == "aic") ? n * std::log(RSS_baseline / n) + 2 * 1 : n * std::log(RSS_baseline / n) + 1 * std::log(n);
    }
    else {
        throw std::invalid_argument(std::format("Unknown threshold {}", threshold_));
    }

    // Keep going until no variable go through the threshold
    std::set<size_t> blacklist;
    bool keep_cond_forward = true;
    std::vector<size_t> features_idx = rangeExcept(p, p);
    while (keep_cond_forward && !features_idx.empty()) {

        // Forward
        // Linear Reg to test each features with the one already selected
        size_t k = x_v.size() / n + 1;
        std::vector<double> criteria;
        criteria.reserve(features_idx.size());
        std::vector<double> v_RSS_model;
        v_RSS_model.reserve(features_idx.size());
        for (auto& feature : features_idx) {

            if (blacklist.count(feature)) continue;
            
            // Adding feature 
            std::vector<double> x_try = x_v;
            x_try.reserve(x_try.size() + n);
            for (size_t i = 0; i < n; i++) {
                x_try.push_back(*X_j[feature][i]);
            }
            
            // Transform to dataframe
            X_try = {n, k, false, std::move(x_try)};

            // Linear Reg
            X_t = ~X_try;  
            X_t.change_layout_inplace();
            XtXInv = (X_t*X_try).inv();
            XtXInv.change_layout_inplace();
            beta_est =  XtXInv * (X_t * y);

            clean_params();
            is_fitted = true;
            coeffs = beta_est.get_data();

            // Stats
            X_try.pop(0);
            double RSS_model = Stats::mse(y.get_data(), predict(X_try)) * n;
            if (threshold_ == "alpha") {
                criteria.push_back((RSS_baseline - RSS_model) / (RSS_model / (n - k)));
            }
            else if (threshold_ == "aic") {
                criteria.push_back(n * std::log(RSS_model / n) + 2 * k);
            }
            else if (threshold_ == "bic") {
                criteria.push_back(n * std::log(RSS_model / n) + k * std::log(n));
            }
            v_RSS_model.push_back(RSS_model);
        }

        // Getting max or min accordingly to the threshold choosens
        // Testing if it go through our threshold
        if (threshold_ == "alpha") {

            // In case of one last value
            size_t idx;
            double p_val;
            if (criteria.size() == 1) {
                keep_cond_forward = false;
            }
            else {
                auto it = std::max_element(criteria.begin(), criteria.end());
                idx = std::distance(criteria.begin(), it);
                p_val = Stats::OLS::fisher_pvalue(criteria[idx], n, n - k); 
                RSS_baseline = v_RSS_model[idx];
            }
            if (keep_cond_forward) {
                if (p_val < alpha_in) {
                    
                    // Adding feature 
                    x_v.reserve(x_v.size() + n);
                    for (size_t i = 0; i < n; i++) {
                        x_v.push_back(*X_j[features_idx[idx]][i]);
                    }

                    features_kept.push_back(features_idx[idx]);
                    features_idx.erase(features_idx.begin() + idx);
                }
                else {
                    keep_cond_forward = false;
                }
            }
        }
        else {
            // In case of one last value
            size_t idx;
            double score_model;
            if (criteria.size() == 1) {
                keep_cond_forward = false;
            }
            else {
                auto it = std::min_element(criteria.begin(), criteria.end());
                idx = std::distance(criteria.begin(), it);
                score_model = criteria[idx];
                RSS_baseline = v_RSS_model[idx];
            }
            if (keep_cond_forward) {
                if ((score_model - score) < 0) {
                    
                    // Adding feature 
                    x_v.reserve(x_v.size() + n);
                    for (size_t i = 0; i < n; i++) {
                        x_v.push_back(*X_j[features_idx[idx]][i]);
                    }

                    features_kept.push_back(features_idx[idx]);
                    features_idx.erase(features_idx.begin() + idx);
                }
                else {
                    keep_cond_forward = false;
                }
            }
        }

        // Calculate our new_score / RSS
        k = x_v.size() / n;
        if (threshold_ == "aic" || threshold_ == "bic") {
            score = (threshold_ == "aic") ? n * std::log(RSS_baseline / n) + 2 * k : n * std::log(RSS_baseline / n) + k * std::log(n);
        }

        // Backward
        std::vector<double> criteria_;
        criteria_.reserve(features_kept.size());

        if ((threshold_ == "aic" || threshold_ == "bic") && k > 2) {
            
            // Linear Reg to test each features with the one selected
            k = x_v.size() / n - 1;
            v_RSS_model.clear();
            v_RSS_model.reserve(features_kept.size());
            std::vector<size_t> feature_backward = rangeExcept(features_kept.size(), features_kept.size());
            for (auto& feature : feature_backward) {
                
                // Erasing a feature 
                std::vector<double> x_try = x_v;
                x_try.erase(x_try.begin() + n * (feature + 1), x_try.begin() + n * (feature + 2));
                
                // Transform to dataframe
                X_try = {n, k, false, std::move(x_try)};

                // Linear Reg
                X_t = ~X_try;  
                X_t.change_layout_inplace();
                XtXInv = (X_t*X_try).inv();
                XtXInv.change_layout_inplace();
                beta_est =  XtXInv * (X_t * y);

                clean_params();
                is_fitted = true;
                coeffs = beta_est.get_data();

                // Stats
                X_try.pop(0);
                double RSS_model = Stats::mse(y.get_data(), predict(X_try)) * n;
                if (threshold_ == "aic") {
                    criteria_.push_back(n * std::log(RSS_model / n) + 2 * k);
                }
                else if (threshold_ == "bic") {
                    criteria_.push_back(n * std::log(RSS_model / n) + k * std::log(n));
                }
                v_RSS_model.push_back(RSS_model);
            }

            // Getting max or min accordingly to the threshold choosens
            auto it = std::min_element(criteria_.begin() + 1, criteria_.end());
            size_t idx = std::distance(criteria_.begin(), it);
            double score_model = criteria_[idx];
            RSS_baseline = v_RSS_model[idx];

            // Testing if it go through our threshold
            if ((score_model - score) < 0) {
                
                // Erasing a feature 
                x_v.erase(x_v.begin() + n * features_kept[idx], x_v.begin() + n * (features_kept[idx] + 1));
                
                blacklist.insert(features_kept[idx]);
                features_idx.insert(features_idx.begin(), features_kept[idx]);
                features_kept.erase(features_kept.begin() + idx - 1);
            }
            else {
                if (!keep_cond_forward) {
                    break;
                }
            }

            // Calculate our new_score / RSS
            k = x_v.size() / n;
            score = (threshold_ == "aic") ? n * std::log(RSS_baseline / n) + 2 * k : n * std::log(RSS_baseline / n) + k * std::log(n);
        }
        
        
        else if (threshold_ == "alpha") {
                    
            // Linear Reg to test each features with the one selected
            k = x_v.size() / n;
            X_try = {n, k, false, x_v};

            // Linear Reg
            X_t = ~X_try;  
            X_t.change_layout_inplace();
            XtXInv = (X_t*X_try).inv();
            XtXInv.change_layout_inplace();
            beta_est =  XtXInv * (X_t * y);

            clean_params();
            is_fitted = true;
            coeffs = beta_est.get_data();

            // Calculating Stats
            X_try.pop(0);
            std::vector<double> residuals = Stats::get_residuals(y.get_data(), predict(X_try));
            std::vector<double> stderr_b = Stats::OLS::stderr_b(residuals, XtXInv);
            std::vector<double> t_stats(k, 0.0);
            for (size_t i = 0; i < k; i++) t_stats[i] = coeffs[i] / stderr_b[i];
            std::vector<double> p_value = Stats::OLS::student_pvalue(t_stats);

            // Getting max or min accordingly to the threshold choosens
            auto it = std::max_element(p_value.begin() + 1, p_value.end());
            size_t idx = std::distance(p_value.begin(), it);
            double p_val = p_value[idx];

            if (p_val > alpha_out) {
                
                // Erasing a feature 
                
                x_v.erase(x_v.begin() + n * features_kept[idx], x_v.begin() + n * (features_kept[idx] + 1));

                blacklist.insert(features_kept[idx]);
                features_idx.insert(features_idx.begin(), features_kept[idx]);
                features_kept.erase(features_kept.begin() + idx - 1);
            }
            else {
                if (!keep_cond_forward) {
                    break;
                }
                keep_cond_forward = true;
            }
        }
    }
    selected_features = features_kept;
    return x_v;
}

std::vector<double> StepwiseRegression::backward_reg(const Dataframe& x, const Dataframe& y) {
    
    // Copy our data 
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    std::vector<double> x_v = x.get_data();
    std::vector<double> features_kept = rangeExcept(static_cast<double>(p), static_cast<double>(p));
    
    // Insert an unit col to get intercept value
    for (size_t i = 0; i < n; i++) {
        x_v.insert(x_v.begin(), 1.0);
    }

    // Need to do a linear reg for aic and bic to get calculate them
    Dataframe X_try = {n, p + 1, false, x_v};

    // Linear Reg
    Dataframe X_t = ~X_try;  
    X_t.change_layout_inplace();
    Dataframe XtXInv = (X_t*X_try).inv();
    XtXInv.change_layout_inplace();
    Dataframe beta_est =  XtXInv * (X_t * y);

    clean_params();
    coeffs = beta_est.get_data();
    is_fitted = true;

    // Variables for threshold
    double score; // AIC or BIC
    double alpha_out;
    std::vector<size_t> features_idx = rangeExcept(p, p);
    std::vector<double> residuals = Stats::get_residuals(y.get_data(), predict(x));
    if (threshold_ == "alpha" && n > 30) {

        std::vector<double> stderr_b = Stats::OLS::stderr_b(residuals, XtXInv);
        std::vector<double> t_stats(p+1, 0.0);
        for (size_t i = 0; i < p+1; i++) t_stats[i] = coeffs[i] / stderr_b[i];
        std::vector<double> p_value = Stats::OLS::student_pvalue(t_stats);
        alpha_out = alpha_.second;

        // Getting max or min accordingly to the threshold choosens
        auto it = std::max_element(p_value.begin() + 1, p_value.end());
        size_t idx = std::distance(p_value.begin(), it);
        double p_val = p_value[idx];
        
        if (p_val > alpha_out) {
            
            // Erasing a feature 
            x_v.erase(x_v.begin() + n * features_idx[idx], x_v.begin() + n * (features_idx[idx] + 1));
            features_idx = rangeExcept(features_idx.size() - 1, features_idx.size() - 1);
            features_kept.erase(features_kept.begin() + idx - 1);
        }
        else {
            selected_features = features_kept;
            return x_v;
        }
    }
    else if (threshold_ == "aic" || threshold_ == "bic") {

        double RSS_baseline = Stats::mse(residuals) * n;
        score = (threshold_ == "aic") ? n * std::log(RSS_baseline / n) + 2 * (p + 1) : n * std::log(RSS_baseline / n) + (p + 1) * std::log(n);
    }
    else {
        throw std::invalid_argument(std::format("Unknown threshold {} or n <= 30 for alpha method", threshold_));
    }

    // Keep going until no variable go through the threshold
    bool keep_cond = true;
    while (keep_cond) {
        std::vector<double> criteria;
        criteria.reserve(features_idx.size());

        if ((threshold_ == "aic" || threshold_ == "bic")) {
            
            // Linear Reg to test each features with the one selected
            size_t k = x_v.size() / n - 1;
            std::vector<double> v_RSS_model;
            v_RSS_model.reserve(features_idx.size());
            for (auto& feature : features_idx) {
                
                // Erasing a feature 
                std::vector<double> x_try = x_v;
                x_try.erase(x_try.begin() + n * (feature + 1), x_try.begin() + n * (feature + 2));
                
                // Transform to dataframe
                X_try = {n, k, false, std::move(x_try)};

                // Linear Reg
                X_t = ~X_try;  
                X_t.change_layout_inplace();
                XtXInv = (X_t*X_try).inv();
                XtXInv.change_layout_inplace();
                beta_est =  XtXInv * (X_t * y);

                clean_params();
                is_fitted = true;
                coeffs = beta_est.get_data();

                // Stats
                X_try.pop(0);
                double RSS_model = Stats::mse(y.get_data(), predict(X_try)) * n;
                if (threshold_ == "aic") {
                    criteria.push_back(n * std::log(RSS_model / n) + 2 * k);
                }
                else if (threshold_ == "bic") {
                    criteria.push_back(n * std::log(RSS_model / n) + k * std::log(n));
                }
                v_RSS_model.push_back(RSS_model);
            }

            // Getting max or min accordingly to the threshold choosens
            auto it = std::min_element(criteria.begin() + 1, criteria.end());
            size_t idx = std::distance(criteria.begin(), it);
            double score_model = criteria[idx];
            double RSS_baseline = v_RSS_model[idx];

            // Testing if it go through our threshold
            if ((score_model - score) < 0) {
                
                // Erasing a feature 
                x_v.erase(x_v.begin() + n * features_idx[idx], x_v.begin() + n * (features_idx[idx] + 1));
                features_idx = rangeExcept(features_idx.size() - 1, features_idx.size() - 1);
                features_kept.erase(features_kept.begin() + idx - 1);
            }
            else {
                keep_cond = false;
                break;
            }

            // Calculate our new_score / RSS
            k = x_v.size() / n;
            score = (threshold_ == "aic") ? n * std::log(RSS_baseline / n) + 2 * k : n * std::log(RSS_baseline / n) + k * std::log(n);
        }
        
        
        else if (threshold_ == "alpha") {
                    
            // Linear Reg to test each features with the ones selected
            size_t k = x_v.size() / n;
            X_try = {n, k, false, x_v};

            // Linear Reg
            X_t = ~X_try;  
            X_t.change_layout_inplace();
            XtXInv = (X_t*X_try).inv();
            XtXInv.change_layout_inplace();
            beta_est =  XtXInv * (X_t * y);

            clean_params();
            is_fitted = true;
            coeffs = beta_est.get_data();

            // Calculating Stats
            X_try.pop(0);
            residuals = Stats::get_residuals(y.get_data(), predict(X_try));
            std::vector<double> stderr_b = Stats::OLS::stderr_b(residuals, XtXInv);
            std::vector<double> t_stats(k, 0.0);
            for (size_t i = 0; i < k; i++) t_stats[i] = coeffs[i] / stderr_b[i];
            std::vector<double> p_value = Stats::OLS::student_pvalue(t_stats);

            // Getting max or min accordingly to the threshold choosens
            auto it = std::max_element(p_value.begin() + 1, p_value.end());
            size_t idx = std::distance(p_value.begin(), it);
            double p_val = p_value[idx];

            if (p_val > alpha_out) {
                
                // Erasing a feature 
                x_v.erase(x_v.begin() + n * features_idx[idx], x_v.begin() + n * (features_idx[idx] + 1));
                features_idx = rangeExcept(features_idx.size() - 1, features_idx[idx]);
                features_kept.erase(features_kept.begin() + idx - 1);
            }
            else {
                keep_cond = false;
                break;
            }
        }
    }
    selected_features = features_kept;
    return x_v;
}

std::vector<double> StepwiseRegression::forward_reg(const Dataframe& x, const Dataframe& y) {
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    std::vector<double> x_v(n, 1.0);
    std::vector<double> features_kept;

    // Getting ptrs to elem of each col for coordinate descent
    std::vector<std::vector<const double*>> X_j(p);
    for (size_t j = 0; j < p; j++) {
        X_j[j] = x.getColumnPtrs(j);
    }

    // Need to do a linear reg with just intercept to begin
    Dataframe X_try = {n, 1, false, x_v};

    // Linear Reg
    Dataframe X_t = ~X_try;  
    X_t.change_layout_inplace();
    Dataframe XtXInv = (X_t*X_try).inv();
    XtXInv.change_layout_inplace();
    Dataframe beta_est =  XtXInv * (X_t * y);

    clean_params();
    coeffs = beta_est.get_data();
    is_fitted = true;

    // Variables for threshold
    double score; // AIC or BIC
    double alpha_in;
    double RSS_baseline = Stats::mse(y.get_data(), predict(X_try)) * n;
    if (threshold_ == "alpha" && n > 30) {
        alpha_in = alpha_.first;
    }
    else if (threshold_ == "aic" || threshold_ == "bic") {
        score = (threshold_ == "aic") ? n * std::log(RSS_baseline / n) + 2 * 1 : n * std::log(RSS_baseline / n) + 1 * std::log(n);
    }
    else {
        throw std::invalid_argument(std::format("Unknown threshold {}", threshold_));
    }

    // Keep going until no variable go through the threshold
    bool keep_cond = true;
    std::vector<size_t> features_idx = rangeExcept(p, p);
    while (keep_cond && !features_idx.empty()) {

        // Linear Reg to test each features with the one already selected
        size_t k = x_v.size() / n + 1;
        std::vector<double> criteria;
        criteria.reserve(features_idx.size());
        std::vector<double> v_RSS_model;
        v_RSS_model.reserve(features_idx.size());
        for (auto& feature : features_idx) {
            
            // Adding feature 
            std::vector<double> x_try = x_v;
            x_try.reserve(x_try.size() + n);
            for (size_t i = 0; i < n; i++) {
                x_try.push_back(*X_j[feature][i]);
            }
            
            // Transform to dataframe
            X_try = {n, k, false, std::move(x_try)};

            // Linear Reg
            X_t = ~X_try;  
            X_t.change_layout_inplace();
            XtXInv = (X_t*X_try).inv();
            XtXInv.change_layout_inplace();
            beta_est =  XtXInv * (X_t * y);

            clean_params();
            is_fitted = true;
            coeffs = beta_est.get_data();

            // Stats
            X_try.pop(0);
            double RSS_model = Stats::mse(y.get_data(), predict(X_try)) * n;
            if (threshold_ == "alpha") {
                criteria.push_back((RSS_baseline - RSS_model) / (RSS_model / (n - k)));
            }
            else if (threshold_ == "aic") {
                criteria.push_back(n * std::log(RSS_model / n) + 2 * k);
            }
            else if (threshold_ == "bic") {
                criteria.push_back(n * std::log(RSS_model / n) + k * std::log(n));
            }
            v_RSS_model.push_back(RSS_model);
        }

        // Getting max or min accordingly to the threshold choosens
        // Testing if it go through our threshold
        if (threshold_ == "alpha") {
            auto it = std::max_element(criteria.begin(), criteria.end());
            size_t idx = std::distance(criteria.begin(), it);
            double p_val = Stats::OLS::fisher_pvalue(criteria[idx], n, n - k); 
            RSS_baseline = v_RSS_model[idx];

            if (p_val < alpha_in) {
                
                // Adding feature 
                x_v.reserve(x_v.size() + n);
                for (size_t i = 0; i < n; i++) {
                    x_v.push_back(*X_j[features_idx[idx]][i]);
                }

                features_kept.push_back(features_idx[idx]);
                features_idx.erase(features_idx.begin() + idx);
            }
            else {
                keep_cond = false;
                break;
            }
        }
        else {
            auto it = std::min_element(criteria.begin(), criteria.end());
            size_t idx = std::distance(criteria.begin(), it);
            double score_model = criteria[idx];
            RSS_baseline = v_RSS_model[idx];

            if ((score_model - score) < 0) {
                
                // Adding feature 
                x_v.reserve(x_v.size() + n);
                for (size_t i = 0; i < n; i++) {
                    x_v.push_back(*X_j[features_idx[idx]][i]);
                }

                features_kept.push_back(features_idx[idx]);
                features_idx.erase(features_idx.begin() + idx);
            }
            else {
                keep_cond = false;
                break;
            }
        }

        // Calculate our new_score / RSS
        k = x_v.size() / n;
        if (threshold_ == "aic" || threshold_ == "bic") {
            score = (threshold_ == "aic") ? n * std::log(RSS_baseline / n) + 2 * k : n * std::log(RSS_baseline / n) + k * std::log(n);
        }
    }
    selected_features = features_kept;
    return x_v;
}

double StepwiseRegression::effective_df() const {
    double df = 0;
    for (size_t i = 1; i < coeffs.size(); i++) {
        if (std::abs(coeffs[i]) > 1e-10) df++;
    }
    return df;
}

void StepwiseRegression::compute_stats(const Dataframe& x, Dataframe& x_const, Dataframe& XtXinv, const Dataframe& y) {
    
    size_t n = x.get_rows();
    size_t p = x_const.get_cols() - 1;
    size_t p_bis = x.get_cols() + 1;
    
    // Degree of liberty
    int df1 = p;
    int df2 = n - df1 - 1;

    // Predict 
    std::vector<double> y_pred = predict(x);
    double nb_features = effective_df();

    // -------------------------------------Calculate stats----------------------------------------
    double r2 = Stats::rsquared(y.get_data(), y_pred);
    double mse = Stats::mse(y.get_data(), y_pred);
    std::vector<double> residuals = Stats::get_residuals(y.get_data(), y_pred);

    // Covariance Matrix of Beta
    std::vector<double> stderr_b = Stats::OLS::stderr_b(residuals, XtXinv);

    double f_stat = Stats::OLS::fisher_test(r2, df1, df2, coeffs, {}, "standard");

    // Add them to our vector of stats
    gen_stats.push_back(r2);

    if (p > 1) gen_stats.push_back(Stats::radjusted(r2, n, p));
    else gen_stats.push_back(-1.0);

    gen_stats.push_back(nb_features);
    gen_stats.push_back(mse);
    gen_stats.push_back(Stats::rmse(mse));
    gen_stats.push_back(Stats::mae(y.get_data(), y_pred));
    gen_stats.push_back(f_stat);
    gen_stats.push_back(Stats::OLS::fisher_pvalue(f_stat, df1, df2));
    gen_stats.push_back(Stats::durbin_watson_test(residuals));
    gen_stats.push_back(Stats::OLS::breusch_pagan_test(x, residuals));

    std::vector<double> resid_stats = Stats::residuals_stats(residuals);
    for (size_t i = 0; i < resid_stats.size(); i++) gen_stats.push_back(resid_stats[i]); 

    if (p > 1) {
        x_const.popup(0);
        std::vector<double> vif = Stats::OLS::VIF(x_const);

        // Mapping
        std::unordered_map<size_t, size_t> feature_to_vif_idx;
        for (size_t i = 0; i < selected_features.size(); i++) {
            feature_to_vif_idx[selected_features[i]] = i;
        }

        gen_stats.push_back(NAN); // intercept
        for (size_t i = 0; i < (p_bis - 1); i++) {
            auto it = feature_to_vif_idx.find(i);
            if (it != feature_to_vif_idx.end()) {
                gen_stats.push_back(vif[it->second]);
            } else {
                gen_stats.push_back(0.0);
            }
        }
    }

    // Update our stderr
    // Mapping
    std::unordered_map<size_t, size_t> beta_idx_to_feature;
    beta_idx_to_feature[0] = 0; // intercept
    for (size_t i = 0; i < selected_features.size(); i++) {
        beta_idx_to_feature[i + 1] = selected_features[i] + 1; // +1 for intercept
    }

    std::vector<double> stderr_b_full(p_bis, 0.0);
    stderr_b_full[0] = stderr_b[0]; // intercept
    for (size_t i = 0; i < selected_features.size(); i++) {
        size_t orig_col = selected_features[i] + 1;
        stderr_b_full[orig_col] = stderr_b[i + 1];
    }
    stderr_b = stderr_b_full;

    // The t-distribution approaches the standard normal distribution for n > 30 
    std::vector<double> p_value;
    std::vector<double> t_stats(p_bis, 0.0);
    if (n > 30) {
        // T_stat
        for (size_t i = 0; i < p_bis; i++) {
            t_stats[i] = (stderr_b[i] != 0.0) ? coeffs[i] / stderr_b[i] : 0.0;
        }
        p_value = Stats::OLS::student_pvalue(t_stats);
    }

    // If we have not the cols name
    std::vector<std::string> headers(p_bis, "");
    headers[0] = "Intercept";
    if (x.get_headers().empty()) {
        for (size_t i = 1; i < p_bis; i++) headers[i] = "c" + std::to_string(i);
    }
    else {
        headers = {"Intercept"};
        headers.insert(headers.end(), x.get_headers().begin(), x.get_headers().end());
    }

    // Save our stats
    CoeffStats c;
    for (size_t i = 0; i < p_bis; i++) {
        if (n > 30) {
            c = {
                headers[i],
                coeffs[i],
                stderr_b[i],
                t_stats[i],
                p_value[i]
            };
        }
        else {
            c = {
                headers[i],
                coeffs[i],
                stderr_b[i],
                NAN,
                NAN
            };
        }
        coeff_stats.push_back(c);
    }
}

void StepwiseRegression::summary(bool detailled) const {
    std::cout << "\n=== REGRESSION SUMMARY ===\n\n";
    
    std::cout << "R2 = " << gen_stats[0] << "\n";
    if (gen_stats[1] != -1.0) std::cout << "Adjusted R2 = " << gen_stats[1] << "\n";
    std::cout << "Eff. DF: " << std::round(gen_stats[2]) << "\n";
    std::cout << "MSE = " << gen_stats[3] << "\n";
    std::cout << "RMSE = " << gen_stats[4] << "\n";
    std::cout << "MAE = " << gen_stats[5] << "\n\n";
    
    std::cout << std::left << std::setw(15) << "Coefficient"
              << std::right << std::setw(15) << "Beta"
              << std::setw(15) << "Stderr"
              << std::setw(15) << "t-stat"
              << std::setw(15) << "p-value"
              << std::setw(5) << "Sig";
    
    if (coeff_stats.size() > 2 && detailled) {
        std::cout << std::setw(15) << "VIF" << "  \n";
        std::cout << std::string(95, '-') << "\n";
    }
    else {
        std::cout << "  \n";
        std::cout << std::string(85, '-') << "\n";
    }

    size_t i = 0;
    for (const auto& stat : coeff_stats) {
        std::cout << std::left << std::setw(15) << stat.name
                  << std::right << std::fixed << std::setprecision(4)
                  << std::setw(15) << stat.beta
                  << std::setw(15) << stat.stderr_beta
                  << std::setw(15) << stat.t_stat
                  << std::setw(15) << stat.p_value
                  << "  " << stat.significance()
                  << std::setw(15);
        
        if (coeff_stats.size() > 2 && detailled) {
            std::cout << std::setw(15) << gen_stats[16 + i] << "\n";
        }
        else {
            std::cout << "\n";
        }
        i++;
    }
    
    std::cout << "\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1\n" << std::endl;

    if (detailled) {
        std::cout << "Additional stats:\n";
        std::cout << "Fisher - F = " << gen_stats[6] << "\n";
        std::cout << "Fisher - p-value = " << gen_stats[7] << "\n";
        std::cout << "Durbin-Watson - rho-value = " << gen_stats[8] << "\n";
        std::cout << "Breusch-Pagan - p-value = " << gen_stats[9] << "\n\n";

        std::cout << "Residuals:\n";
        std::cout << std::right << std::fixed
                << std::setw(15) << "Mean" 
                << std::setw(15) << "Stdd"
                << std::setw(15) << "Abs Max"
                << std::setw(15) << "Q1"
                << std::setw(15) << "Q2"
                << std::setw(15) << "Q3" << "\n";
        std::cout << std::setw(90) << std::setfill('-') << "" << std::setfill(' ') << "\n";
        std::cout << std::right << std::fixed << std::setprecision(4)
                << std::setw(15) << gen_stats[10]
                << std::setw(15) << gen_stats[11]
                << std::setw(15) << gen_stats[12]
                << std::setw(15) << gen_stats[13]
                << std::setw(15) << gen_stats[14]
                << std::setw(15) << gen_stats[15] << "\n" << std::endl;
    }
}
}