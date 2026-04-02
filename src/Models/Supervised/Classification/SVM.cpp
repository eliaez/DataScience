#include <iomanip>
#include <iostream>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Stats/stats_reg.hpp"
#include "Stats/stats_class.hpp"
#include "Models/Supervised/Classification/SVM.hpp"

using namespace Utils;

namespace Class {

Dataframe SVM_Algo::kernel_meth(const Dataframe& X1, const Dataframe& X2) const {
    
    Dataframe K;
    if (kernel_ == "linear") {
        K = X1 * X2;
        return K;
    }

    size_t n1 = X1.get_rows();
    size_t p = X1.get_cols();
    size_t n2 = X2.get_cols();
    std::vector<double> inter = (X1 * X2).get_data();

    if (kernel_ == "poly") {
        std::vector<double> c(n1 * n2, 1.0);
        inter = mult(inter, gamma_val);
        inter = add(inter, c);
        inter = pow_vect(inter, degree_);
    }
    else if (kernel_ == "rbf") {
        std::vector<double> d_sv(n1, 0.0), d_pred(n2, 0.0);
        for (size_t i = 0; i < n1; i++)
            for (size_t d = 0; d < p; d++)
                d_sv[i] += X1.at(i*p+d) * X1.at(i*p+d); // Row major
        
        for (size_t j = 0; j < n2; j++)
            for (size_t d = 0; d < p; d++)
                d_pred[j] += X2.at(j*p+d) * X2.at(j*p+d); // Col major

        // Create our matrix D col major
        for (size_t j = 0; j < n2; j++)
            for (size_t i = 0; i < n1; i++)
                inter[j*n1+i] = std::exp(-gamma_val * (d_sv[i] + d_pred[j] - 2*inter[j*n1+i]));
    }
    else throw std::invalid_argument("Unknown kernel: " + kernel_);

    K = {n1, n2, false, std::move(inter)};
    return K;
}

Dataframe SVM_Algo::fit_without_stats(const Dataframe& x, const Dataframe& y) {
    
    // Tests
    basic_verif(x);
    basic_verif(y);
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }

    nb_categories(y);
    if (y.get_cols() > 1 || nb_cats > 2) {
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

    // Convert labels 0 to -1
    std::vector<double> y_v = y.get_data();
    for (size_t i = 0; i < n; i++) if (y_v[i] == 0) y_v[i] = -1;

    // Compute gamma_val 
    if (gamma_ == "scale") gamma_val = 1 / (p * Stats::var(X.get_data()) * (n * p - 1) / (n * p));
    else if (gamma_ == "auto") gamma_val = 1 / p;
    else throw std::invalid_argument("Unknown gamma: " + gamma_);

    // Construct our Kernel matrix
    Dataframe K = kernel_meth(X, X_T);

    // SMO (Sequential Minimal Optimization)
    int idx = 0;
    double b = 0;
    K.is_symmetric();
    int numChanged = 0;
    bool examineAll = true;
    std::vector<double> alpha(n, 0.0);

    // Init errors
    std::vector<double> errors(n, 0.0);
    for (size_t i = 0; i < n; i++) errors[i] = -y_v[i];

    // Get error (recompute if bounded)
    auto get_error = [&](size_t i) -> double {
        double a = alpha[i];
        if (a > 1e-8 && a < C_ - 1e-8) return errors[i];
        double val = b;
        for (size_t j = 0; j < n; j++)
            val += alpha[j] * y_v[j] * K.at(i * n + j);
        return val - y_v[i];
    };
    
    while ((numChanged > 0 || examineAll) && idx < max_iter_) {
        numChanged = 0;
        
        for (size_t i = 0; i < n; i++) {

            size_t m = i;
            double alpha_m = alpha[m];
            if (!examineAll && (alpha_m <= 1e-8 || alpha_m >= C_ - 1e-8)) continue;
            
            double error_m = get_error(m);

            // KKT conditions
            double val = 0.0;
            if (alpha_m <= 1e-8) val = std::max(0.0, - y_v[m] * error_m);
            else if (alpha_m >= C_ - 1e-8) val = std::max(0.0, y_v[m] * error_m);
            else val = std::abs( y_v[m] * error_m);

            if (val < tol_) continue;

            // Choose o with argmin/argmax
            size_t o = n;
            
            if (error_m > 0.0) {
                double best = std::numeric_limits<double>::max();
                for (size_t j = 0; j < n; j++) {
                    if (alpha[j] <= 1e-8 || alpha[j] >= C_ - 1e-8) continue;
                    if (errors[j] < best || o == n) { 
                        best = std::abs(error_m - errors[j]);
                        o = j; 
                    }
                }
            } else {
                double best = std::numeric_limits<double>::lowest();
                for (size_t j = 0; j < n; j++) {
                    if (alpha[j] <= 1e-8 || alpha[j] >= C_ - 1e-8) continue;
                    if (errors[j] > best || o == n) { 
                        best = std::abs(error_m - errors[j]);
                        o = j; 
                    }
                }
            }

            // Fallback for o
            if (o == n) {

                size_t start = std::rand() % n;
                for (size_t k = 0; k < n; k++) {

                    size_t j = (start + k) % n;
                    if (j != m) { 
                        o = j; 
                        break; 
                    }
                }
            }
            if (o == n || o == m) continue;

            // Calculate bounds L, H
            double L = 0.0;
            double H = 0.0;
            double alpha_o = alpha[o];
            if (y_v[m] == y_v[o]) {
                L = std::max(0.0, alpha_o + alpha_m - C_);
                H = std::min(C_, alpha_o + alpha_m);
            }
            else {
                L = std::max(0.0, alpha_m - alpha_o);
                H = std::min(C_, C_ + alpha_m - alpha_o);
            }
            if (H - L < 1e-12) continue;

            // Calculate eta
            double eta = K.at(m * n + m) + K.at(o * n + o) - 2 * K.at(m * n + o);
            if (eta <= 0) continue;

            double error_o = get_error(o);

            // Calculate and clip alpha_o
            double alpha_m_new = alpha_m + y_v[m] * (error_o - error_m) / eta;
            if (alpha_m_new < L) alpha_m_new = L;
            else if (alpha_m_new > H) alpha_m_new = H;
            if (std::abs(alpha_m_new - alpha_m) < 1e-8 * (alpha_m_new + alpha_m + 1e-8)) continue;

            // Calculate alpha_m and clip it
            double alpha_o_new = alpha_o + y_v[m] * y_v[o] * (alpha_m - alpha_m_new);
            if (alpha_o_new < 1e-8) alpha_o_new = 0.0;
            else if (alpha_o_new > C_ - 1e-8) alpha_o_new = C_;

            double delta_m = alpha_m_new - alpha_m;
            double delta_o = alpha_o_new - alpha_o;

            // Update b 
            double bm = b - error_m - y_v[m] * delta_m * K.at(m*n+m) - y_v[o] * delta_o * K.at(m*n+o);
            double bo = b - error_o - y_v[m] * delta_m * K.at(m*n+o)  - y_v[o] * delta_o * K.at(o*n+o);

            double old_b = b;
            if (alpha_m_new > 0 && alpha_m_new < C_) b = bm;
            else if (alpha_o_new > 0 && alpha_o_new < C_) b = bo;
            else b = (bm + bo) / 2.0;

            // Update our alpha with alpha_m and alpha_o
            alpha[m] = alpha_m_new;
            alpha[o] = alpha_o_new;

            // Update our errors (only for non bounds ones)
            errors[m] = 0.0;
            errors[o] = 0.0;
            double db = b - old_b;
            for (size_t j = 0; j < n; j++) {

                if (j == o || j == m) continue;
                if (alpha[j] > 0.0 && alpha[j] < C_) {
                    double delta = y_v[m] * delta_m * K.at(j * n + m) + y_v[o] * delta_o * K.at(j * n + o);
                    errors[j] += delta + db;
                }
            }
            numChanged++;
        }

        if (examineAll) examineAll = false;
        else if (numChanged == 0) examineAll = true;
        idx++;
    }
    if (max_iter_ == idx) 
        std::cout << "Max_iter reached, you might need to change max_iter or scale your data\n" << std::endl;

    // Calculate final W or getting alpha 
    if (kernel_ == "linear") {
        std::vector<double> inter(n, 0.0);
        for (size_t i = 0; i < n; i++) inter[i] = alpha[i] * y_v[i];
        Dataframe alpha_y = {n, 1, false, std::move(inter)};
        Dataframe W = X_T_row * alpha_y;

        // Results
        coeffs = W.get_data();
        coeffs.insert(coeffs.begin(), b);
    }
    else {
        sv_x.clear();
        sv_alpha_y.clear();

        for (size_t i = 0; i < n; i++) {
            if (alpha[i] > 1e-8) {
                for (size_t j = 0; j < p; j++) sv_x.push_back(x.at(j * n + i));
                sv_alpha_y.push_back(alpha[i] * y_v[i]);
            }
        }
        coeffs = {b};
    }

    alpha_ = std::move(alpha);
    sv_bool.assign(n, false);
    for (size_t i = 0; i < n; i++) if (alpha_[i] > 1e-8) sv_bool[i] = true;
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

    Dataframe Y;
    if (kernel_ == "linear") {

        // Insert an unit col for intercept value
        x_v.insert(x_v.begin(), n, 1.0);
        Dataframe X = {n, p+1, false, std::move(x_v)};
        X.change_layout_inplace();

        // Calculate y_pred
        Dataframe W = {p+1, 1, false, coeffs};
        Y = X * W;
    }
    else {
        size_t n_sv = sv_alpha_y.size();
        
        Dataframe X_sv = {n_sv, p, true, sv_x};
        Dataframe X = {n, p, false, std::move(x_v)};
        Dataframe X_T = ~X;
        Dataframe K = kernel_meth(X_sv, X_T);

        K = ~K;
        K.change_layout_inplace();
        Dataframe SV_alpha_y = {n_sv, 1, false, sv_alpha_y};
        
        // b
        std::vector<double> b_v(n, coeffs[0]);
        Dataframe B = {n, 1, false, std::move(b_v)};

        Y = (K * SV_alpha_y) + B;
    }

    std::vector<double> y_pred;
    y_pred.reserve(n);
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
    Dataframe Y_pred = {n, 1, false, std::move(y_pred)};
    
    // Confusion matrix
    std::vector<double> conf_matrix = Stats_class::conf_matrix(y.get_data(), Y_pred.get_data());

    // Roc Auc 
    double roc_auc = Stats_class::roc_auc(y.get_data(), Y_pred.get_data());

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

    int n_sv = std::count(sv_bool.begin(), sv_bool.end(), true);
    
    if (kernel_ == "linear") {
        std::vector<double> w(coeffs.begin() + 1, coeffs.end());
        double w_norm = Lnorm(w, 2);
        gen_stats.push_back(2.0 / w_norm);
    } 
    else {
        Dataframe SV_alpha_y_T = {1, static_cast<size_t>(n_sv), true, sv_alpha_y};
        Dataframe X_sv = {static_cast<size_t>(n_sv), p, true, sv_x};
        Dataframe X_sv_T = ~X_sv;
        X_sv.change_layout_inplace();

        Dataframe K = kernel_meth(X_sv, X_sv_T);
        std::vector<double> W_norm = (SV_alpha_y_T * K).get_data();

        double w_norm_sq = dot(W_norm, sv_alpha_y); 
        gen_stats.push_back(2.0 / std::sqrt(w_norm_sq));
    }

    gen_stats.push_back(n_sv);
    gen_stats.push_back(static_cast<double>(n_sv) * 100.0 / n);
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
    std::cout << "% of SV     = " << gen_stats[2] << "%\n";
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
                    << std::right << std::fixed << std::setprecision(4);
        
        if (kernel_ == "linear" || i == 0)
            std::cout << std::setw(12) << stat.beta[i] << "\n";
        else
            std::cout << std::setw(12) << " _ \n";
    }
    std::cout << "\n" << std::endl;
}
}