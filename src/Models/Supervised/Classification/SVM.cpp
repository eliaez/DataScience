#include <iomanip>
#include <iostream>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Stats/stats_reg.hpp"
#include "Stats/stats_class.hpp"
#include "Models/Supervised/Classification/SVM.hpp"

using namespace Utils;

namespace Class {

Dataframe SVM_Algo::fit_without_stats(const Dataframe& x, const Dataframe& y) {
    
    // Tests
    basic_verif(x);
    basic_verif(y);
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }

    nb_categories(y);
    if (y.get_cols() > 2 || nb_cats > 2) {
        throw std::invalid_argument("SVM support only binary classfication");
    }

    size_t n = x.get_rows();
    size_t p = x.get_cols();

    // Creating appropriate Dataframe
    std::vector<double> x_v = x.get_data();
    Dataframe X = {n, p, false, std::move(x_v)};
    Dataframe X_T = ~X;
    Dataframe X_T_row = X_T.change_layout();
    X.change_layout_inplace();

    // Construct our Kernel matrix
    Dataframe K;
    if (kernel_ == "linear") {
        K = X * X_T;
    }
    else if (kernel_ == "poly") {
        std::vector<double> inter = (X * X_T).get_data();
        std::vector<double> c(n * n, 1.0);
        inter = add(inter, c);
        inter = pow_vect(inter, degree_);
        K = {n, n, false, std::move(inter)};
    }
    else if (kernel_ == "rbf") {
        std::vector<double> inter = (X * X_T).get_data();
        
        std::vector<double> diag;
        diag.reserve(n);
        for (size_t i = 0; i < n; i++) diag.push_back(inter[i * n + i]); 

        if (gamma_ == "scale") gamma_val = 1 / (p * Stats::var(X.get_data()));
        else if (gamma_ == "auto") gamma_val = 1 / p;
        else throw std::invalid_argument("Unknown gamma: " + gamma_);

        // TODO AVX2
        // Create our matrix D
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++)
                inter[i * n + j] = std::exp(- gamma_val * (diag[i] + diag[j] - 2 * inter[i * n + j]));
        }
        K = {n, n, false, std::move(inter)};
    }
    else if (kernel_ == "sigmoid") {
        std::vector<double> inter = (X * X_T).get_data();
        std::vector<double> c(n * n, 1.0);

        if (gamma_ == "scale") gamma_val = 1 / (p * Stats::var(X.get_data()));
        else if (gamma_ == "auto") gamma_val = 1 / p;
        else throw std::invalid_argument("Unknown gamma: " + gamma_);

        inter = mult(inter, gamma_val);
        inter = add(inter, c);
        for (auto& val : inter) val = std::tanh(val);
        K = {n, n, false, std::move(inter)};
    }
    else throw std::invalid_argument("Unknown kernel: " + kernel_);

    // SMO (Sequential Minimal Optimization)
    int idx = 0;
    double b = 0;
    K.is_symmetric();
    bool keep_cond = true;
    std::vector<double> errors;
    std::vector<double> alpha(n, 0.0);
    while (keep_cond && idx < max_iter_) {

        // TODO AVX2
        // Calculate errors and choose m with KKT conditions
        size_t m = 0;
        errors.clear();
        errors.reserve(n);
        double cond_value = 0.0;
        for (size_t i = 0; i < n; i++) {

            // Errors part
            double error = b - y.at(i);
            for (size_t j = 0; j < n; j++) {
                error += alpha[j] * y.at(j) * K.at(i * n + j);
            }
            errors.push_back(error);

            // KKT condition 
            double val = 0.0;
            if (alpha[i] == 0) val = std::max(0.0, - y.at(i) * errors[i]);
            else if (alpha[i] == C_) val = std::max(0.0, y.at(i) * errors[i]);
            else val = std::abs( y.at(i) * errors[i]);

            if (val > cond_value) {
                cond_value = val;
                m = i;
            }
        }

        // Choose o with KKT
        size_t o = 0;
        cond_value = 0.0;
        double error_m = errors[m];
        for (size_t i = 0; i < n; i++) {
            
            double val = std::abs(error_m - errors[i]);
            if ( val > cond_value && i != m) {
                cond_value = val;
                o = i;
            } 
        }

        // Calculate bounds L, H and eta 
        double L = 0.0;
        double H = 0.0;
        if (y.at(m) == y.at(o)) {
            L = std::max(0.0, alpha[o] + alpha[m] - C_);
            H = std::min(C_, alpha[o] + alpha[m]);
        }
        else {
            L = std::max(0.0, alpha[o] - alpha[m]);
            H = std::min(C_, C_ + alpha[o] - alpha[m]);
        }

        // Calculate eta
        double eta = K.at(m * n + m) + K.at(o * n + o) - 2 * K.at(m * n + o);
        if (eta <= 0) { 
            idx++; 
            continue; 
        }

        // Calculate and clip alpha_o
        double alpha_o_new = alpha[o] + y.at(o) * (errors[m] - errors[o]) / eta;
        if (alpha_o_new < L) alpha_o_new = L;
        else if (alpha_o_new > H) alpha_o_new = H;

        // Calculate alpha_m
        double alpha_m_new = alpha[m] + y.at(m) * y.at(o) * (alpha[o] - alpha_o_new);

        double delta_m = alpha_m_new - alpha[m];
        double delta_o = alpha_o_new - alpha[o];

        // Update b 
        if (alpha_m_new < C_ && alpha_m_new > 0) {
            double bm = b - errors[m] - y.at(m) * delta_m * K.at(m * n + m) - y.at(o) * delta_o * K.at(m * n + o);
            b = bm;
        }
        else {
            if (alpha_o_new < C_ && alpha_o_new > 0) {
                double bo = b - errors[o] - y.at(m) * delta_m * K.at(m * n + o) - y.at(o) * delta_o * K.at(o * n + o);
                b = bo;
            }
            else {
                double bm = b - errors[m] - y.at(m) * delta_m * K.at(m * n + m) - y.at(o) * delta_o * K.at(m * n + o);
                double bo = b - errors[o] - y.at(m) * delta_m * K.at(m * n + o) - y.at(o) * delta_o * K.at(o * n + o);
                b = (bm + bo) / 2;
            }
        }

        // Update our alpha with alpha_m and alpha_o
        alpha[m] = alpha_m_new;
        alpha[o] = alpha_o_new;

        // TODO AVX2
        // Update our errors
        for (size_t i = 0; i < n; i++) {
            double delta = y.at(m) * delta_m * K.at(i * n + m) + y.at(o) * delta_o * K.at(i * n + o);
            errors[i] += delta;
        }
        
        // Convergence
        keep_cond = false;
        for (size_t i = 0; i < n; i++) {

            double yE = y.at(i) * errors[i];
            if (alpha[i] < C_ && yE < -tol_) {
                keep_cond = true;
                break;
            }
            if (alpha[i] > 0 && yE > tol_) {
                keep_cond = true;
                break;
            }
        }
        idx++;
    }
    if (max_iter_ == idx) 
        std::cout << "Max_iter reached, you might need to change learning_rate or scale your data\n" << std::endl;

    // Calculate final W
    std::vector<double> inter;
    inter.reserve(n);
    for (size_t i = 0; i < n; i++) inter.push_back(alpha[i] * y.at(i));
    Dataframe alpha_y = {1, n, false, std::move(inter)};
    Dataframe W = X_T_row * alpha_y;

    // Results
    coeffs = W.get_data();
    coeffs.insert(coeffs.begin(), b);
    alpha_ = std::move(alpha);
    support_vector.assign(n, false);
    for (size_t i = 0; i < n; i++) {
        if (alpha_[i] > 0 && alpha_[i] < C_)
            support_vector[i] = true;
    }
    is_fitted = true;

    return {};
}

std::vector<double> SVM_Algo::predict(const Dataframe& x) const {
    basic_verif(x);
    if (!is_fitted) {
        throw std::runtime_error("Need to have trained your model");
    }
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }
    size_t n = x.get_rows();
    size_t p = x.get_cols();

    // Copy our data 
    std::vector<double> x_v = x.get_data();
    
    // Insert an unit col for intercept value
    x_v.insert(x_v.begin(), n, 1.0);
    Dataframe X = {n, p+1, false, std::move(x_v)};
    X.change_layout_inplace();

    // Calculate y_pred
    std::vector<double> y_pred;
    y_pred.reserve(n);
    Dataframe W = {1, p+1, false, coeffs};
    Dataframe Y = X * W;
    for (size_t i = 0; i < n; i++) y_pred.push_back(Y.at(i) > 0 ? 1.0 : 0.0);

    return y_pred;
}

std::unique_ptr<ClassificationBase> SVM_Algo::create(const std::vector<std::variant<double, std::string>>& params) {

    if (params.size() > 3) {
        return std::make_unique<SVM_Algo>(
            std::get<double>(params[0]), 
            std::get<std::string>(params[1]),
            std::get<std::string>(params[2]),
            std::get<double>(params[3])
        );
    }
    else if (params.size() > 2) {
        return std::make_unique<SVM_Algo>(
            std::get<double>(params[0]), 
            std::get<std::string>(params[1]),
            std::get<std::string>(params[2])
        );
    }
    return std::make_unique<SVM_Algo>(std::get<double>(params[0]), std::get<std::string>(params[1]));
}

void SVM_Algo::compute_stats(const Dataframe& x, Dataframe& /*x_const*/, const Dataframe& y) {
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    
    // Predict 
    std::vector<double> y_pred = predict(x);
    Dataframe Y_pred = {n, 2, false, std::move(y_pred)};
    
    // Confusion matrix
    std::vector<double> conf_matrix = Stats_class::conf_matrix(y.get_data(), Y_pred.get_data());

    // Roc Auc 
    double roc_auc = Stats_class::roc_auc(y.get_data(), y_pred);

    // If we have not the cols name
    std::vector<std::string> headers(p+1, "");
    headers[0] = "Intercept";
    if (x.get_headers().empty()) {
        for (size_t i = 1; i < p+1; i++) headers[i] = "c" + std::to_string(i);
    }
    else {
        headers = {"Intercept"};
        headers.insert(headers.end(), x.get_headers().begin(), x.get_headers().end());
    }

    // Coeff stats
    CoeffStats c;
    c.category = "";
    for (size_t i = 0; i < p+1; i++) {
        c.name.push_back(headers[i]);
        c.beta.push_back(coeffs[i]);
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

    std::vector<double> w(coeffs.begin() + 1, coeffs.end());
    double w_norm = Lnorm(w, 2); 
    int n_sv = std::count(support_vector.begin(), support_vector.end(), true);

    gen_stats.push_back(2.0 / w_norm);
    gen_stats.push_back(n_sv);
    gen_stats.push_back(n_sv / n);
    gen_stats.push_back(Stats_class::mcc(conf_matrix));
    gen_stats.push_back((TP + TN) / n);
    gen_stats.push_back(prec);
    gen_stats.push_back(rec);
    gen_stats.push_back(spec);
    gen_stats.push_back(f1_);
    gen_stats.push_back(roc_auc);
}

void SVM_Algo::summary(bool detailled) const {

    std::cout << "\n=== Classification SUMMARY ===\n\n";

    // Global Stats
    std::cout << "Margin      = " << gen_stats[0] << "\n";
    std::cout << "Nb of SV    = " << gen_stats[1] << "\n";
    std::cout << "% of SV     = " << gen_stats[2] << "\n";
    std::cout << "MCC         = " << gen_stats[3] << "\n";
    std::cout << "Accuracy    = " << gen_stats[4] << "\n\n";

    if (detailled) {
        std::cout << "Precision   = " << gen_stats[5] << "\n";
        std::cout << "Recall      = " << gen_stats[6] << "\n";
        std::cout << "Specificity = " << gen_stats[7] << "\n";
        std::cout << "F1          = " << gen_stats[8] << "\n";
        std::cout << "ROC AUC     = " << gen_stats[9] << "\n\n";
    }

    std::cout << "---  Coefficient  ---\n";

    // Coefficients table
    std::cout << std::left  << std::setw(25) << "Feature"
                << std::right << std::setw(12) << "Weights" << "\n";
    std::cout << std::string(40, '-') << "\n";

    CoeffStats stat = coeff_stats[0];
    for (size_t i = 0; i < stat.name.size(); i++) {
        std::cout << std::left  << std::setw(25) << stat.name[i]
                    << std::right << std::fixed << std::setprecision(4)
                    << std::setw(12) << stat.beta[i] << "\n";
    }
    std::cout << "\n" << std::endl;
}
}