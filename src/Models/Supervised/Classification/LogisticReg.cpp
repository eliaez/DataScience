#include <iomanip>
#include <iostream>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Stats/stats_reg.hpp"
#include "Stats/stats_class.hpp"
#include "Validation/Validation.hpp"
#include "Models/Supervised/Classification/LogisticReg.hpp"

using namespace Utils;

namespace Class {

std::pair<Dataframe, Dataframe> LogisticRegression::fit_without_stats(const Dataframe& x, const Dataframe& y) {
    
    // Tests
    basic_verif(x);
    basic_verif(y);
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }

    nb_categories(y);
    if (y.get_cols() != nb_cats && nb_cats > 2 && y.get_cols() != 1) {
        throw std::invalid_argument("For Y, either input Y col major with nb of cols = nb of categories or only one col");
    }
    else if (y.get_cols() == nb_cats && nb_cats > 2 && y.get_storage()) {
        throw std::invalid_argument("For Y, if you input Y with nb of cols = nb of categories then Y need to be col major");
    }

    size_t n = x.get_rows();
    size_t p = x.get_cols();

    // Copy our data 
    std::vector<double> x_v = x.get_data();
    
    // Insert an unit col to get intercept value
    for (size_t i = 0; i < n; i++) {
        x_v.insert(x_v.begin(), 1.0);
    }
    
    // Need X
    Dataframe X = {n, p+1, false, std::move(x_v)};
    Dataframe X_T = ~X;
    X_T.change_layout_inplace();

    // Our vector W
    std::vector<double> w(p+1, 0.0);
    Dataframe W = {nb_cats, p+1, false, std::move(w)};

    // Gradient Descent
    int idx = 0;
    Dataframe Y_ = y;
    double loss = 0.0;
    bool keep_cond = true;
    double old_loss = -1.0;
    if (nb_cats > 2 && y.get_cols() == 1) Y_.OneHot(0);
    while (keep_cond && idx < max_iter_) {

        // Softmax
        std::vector<double> y_v = softmax(X, W);
        Y_ = Dataframe(n, nb_cats, false, std::move(y_v));

        // Cost function 
        loss = Stats_class::cat_logloss(y.get_data(), Y_.get_data(), nb_cats);

        if (penality_ == 1.0) {
            for (size_t i = 0; i < (p+1)*nb_cats; i++) {

                // To exclude w0
                if (i % (p+1) == 0) continue;
                loss += std::abs(W.at(i)) / C_;
            }
        }
        else if (penality_ == 2.0) {
            for (size_t i = 0; i < (p+1)*nb_cats; i++) {

                // To exclude w0
                if (i % (p+1) == 0) continue;
                loss += W.at(i) * W.at(i) / (2 * C_);
            }
        }
        else {
            throw std::invalid_argument("Unknown penality: " + std::to_string(penality_));
        }

        // Calculate our gradient
        std::vector<double> gradient_v = (X_T * (Y_ - y)).get_data();
        gradient_v = mult(gradient_v, learning_r_ / n);
        Dataframe gradient = {p+1, nb_cats, false, std::move(gradient_v)};

        // Update our W
        W = W - gradient;

        // Testing convergence of cost
        if (std::abs(loss - old_loss) < tol_) { // Threshold
            break;
        }
        old_loss = loss;
        idx++;
    }

    // Results
    coeffs = W.get_data();
    is_fitted = true;

    return {X, X_T};
}

/*
void LogisticRegression::optimal_lambda(double start, double end, int nb, const Dataframe& x, const Dataframe& y) {
    std::vector<double> path(nb);
    double log_min = log(start);
    double log_max = log(end);
    double step = (log_max - log_min) / (nb - 1);

    for (int i = 0; i < nb; i++) {
        path[i] = exp(log_min + i * step);
    }

    std::vector<std::vector<double>> param_grid = {path};
    Validation::GSres res = Validation::GSearchCV(this, x, y, param_grid);

    lambda_ = res.best_params[0];
}*/

std::unique_ptr<ClassificationBase> LogisticRegression::create(const std::vector<double>& params) {
    return std::make_unique<LogisticRegression>(params[0], params[1]);
}

void LogisticRegression::compute_stats(const Dataframe& x, Dataframe& x_const, const Dataframe& X_T, const Dataframe& y) {
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    
    // Predict 
    std::vector<double> y_pred = predict(x);

    // -------------------------------------Calculate stats----------------------------------------
    double r2 = Stats::rsquared(y.get_data(), y_pred);
    double mse = Stats::mse(y.get_data(), y_pred);
    std::vector<double> residuals = Stats::get_residuals(y.get_data(), y_pred);

    // Covariance Matrix of Beta
    Dataframe cov_beta = Stats::OLS::cov_beta(x_const, XtXinv, residuals, cov_type_, cluster_ids_);
    std::vector<double> stderr_b = Stats::OLS::stderr_b(cov_beta);

    double f_stat = Stats::OLS::fisher_test(r2, df1, df2, coeffs, cov_beta, cov_type_);

    // Add them to our vector of stats
    gen_stats.push_back(r2);

    if (p > 1) gen_stats.push_back(Stats::radjusted(r2, n, p));
    else gen_stats.push_back(-1.0);

    gen_stats.push_back(mse);
    gen_stats.push_back(Stats::rmse(mse));
    gen_stats.push_back(Stats::mae(y.get_data(), y_pred));
    gen_stats.push_back(f_stat);
    gen_stats.push_back(Stats::OLS::fisher_pvalue(f_stat, df1, df2));
    gen_stats.push_back(Stats::durbin_watson_test(residuals));
    gen_stats.push_back(Stats::OLS::breusch_pagan_test(x, residuals));

    std::vector<double> resid_stats = Stats::residuals_stats(residuals);
    for (size_t i = 0; i < resid_stats.size(); i++) gen_stats.push_back(resid_stats[i]); 

    // The t-distribution approaches the standard normal distribution for n > 30 
    std::vector<double> p_value;
    std::vector<double> t_stats(p+1, 0.0);
    if (n > 30) {
        for (size_t i = 0; i < p+1; i++) t_stats[i] = coeffs[i] / stderr_b[i];
        p_value = Stats::OLS::student_pvalue(t_stats);
    }

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

    // Save our stats
    CoeffStats c;
    for (size_t i = 0; i < p+1; i++) {
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

void LogisticRegression::summary(bool detailled) const {
    std::cout << "\n=== REGRESSION SUMMARY ===\n\n";
    
    std::cout << "R2 = " << gen_stats[0] << "\n";
    if (gen_stats[1] != -1.0) std::cout << "Adjusted R2 = " << gen_stats[1] << "\n";
    std::cout << "MSE = " << gen_stats[2] << "\n";
    std::cout << "RMSE = " << gen_stats[3] << "\n";
    std::cout << "MAE = " << gen_stats[4] << "\n\n";
    
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
                  << std::setw(15) << stat.z_stat
                  << std::setw(15) << stat.p_value
                  << "  " << stat.significance()
                  << std::setw(15);
        
        if (coeff_stats.size() > 2 && detailled) {
            std::cout << std::setw(15) << gen_stats[15 + i] << "\n";
        }
        else {
            std::cout << "\n";
        }
        i++;
    }
    
    std::cout << "\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1\n" << std::endl;

    if (detailled) {
        std::cout << "Additional stats:\n";
        std::cout << "Fisher - F = " << gen_stats[5] << "\n";
        std::cout << "Fisher - p-value = " << gen_stats[6] << "\n";
        std::cout << "Durbin-Watson - rho-value = " << gen_stats[7] << "\n";
        std::cout << "Breusch-Pagan - p-value = " << gen_stats[8] << "\n\n";

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
                << std::setw(15) << gen_stats[9]
                << std::setw(15) << gen_stats[10]
                << std::setw(15) << gen_stats[11]
                << std::setw(15) << gen_stats[12]
                << std::setw(15) << gen_stats[13]
                << std::setw(15) << gen_stats[14] << "\n" << std::endl;
    }
}
}