#include <iomanip>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Linalg/Linalg.hpp"
#include "Stats/stats_reg.hpp"
#include "Models/Supervised/Regression/LinearReg.hpp"

using namespace Utils;

namespace Reg {

void LinearRegression::fit(const Dataframe& x, const Dataframe& y) {
    
    auto [x_const, XtXInv] = fit_without_stats(x, y);
    compute_stats(x, x_const, XtXInv, y);
}

std::pair<Dataframe, Dataframe> LinearRegression::fit_without_stats(const Dataframe& x, const Dataframe& y) {
    
    // Tests
    basic_verif(x);
    basic_verif(y);
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }

    size_t n = x.get_rows();
    size_t p = x.get_cols();

    // In case of GLS
    Dataframe Om;
    bool is_gls = false; 
    if (Omega_ != nullptr && !Omega_->get_data().empty()) {
        is_gls = true;
        basic_verif((*Omega_));
        
        if (Omega_->get_cols() != n) {
            throw std::invalid_argument("Omega need to have n rows and cols with n = x.rows");
        }
        Om = Omega_->inv();
    }

    // Copy our data 
    std::vector<double> x_v = x.get_data();
    
    // Insert an unit col to get intercept value
    for (size_t i = 0; i < n; i++) {
        x_v.insert(x_v.begin(), 1.0);
    }

    // Need X col-major (for mult ops)
    Dataframe X = {n, p+1, false, std::move(x_v)};

    // Need X_t row-major (for mult ops)
    Dataframe X_t = ~X;  // Transpose change it to col-major
    X_t.change_layout_inplace();

    // Calculate Beta (our estimator) for GLS or classical OLS
    Dataframe XtXInv;
    Dataframe beta_est;
    if (is_gls) {
        Dataframe XtOmega = X_t * Om;
        XtOmega.change_layout_inplace();
        XtXInv =  (XtOmega * X).inv();
        XtXInv.change_layout_inplace();    
        beta_est =  XtXInv * (XtOmega * y);
    }
    else {
        XtXInv = (X_t*X).inv();
        XtXInv.change_layout_inplace();
        beta_est =  XtXInv * (X_t * y);  
    }

    // Results
    coeffs = beta_est.get_data();
    is_fitted = true;

    return {X, XtXInv};
}

void LinearRegression::compute_stats(const Dataframe& x, const Dataframe& x_const, Dataframe& XtXinv, const Dataframe& y) {
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    
    // Degree of liberty
    int df1 = p;
    int df2 = n - df1 - 1;

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

    if (p > 1) {
        std::vector<double> vif = Omega_ == nullptr ? Stats::OLS::VIF(x) : Stats::OLS::VIF(x, (*Omega_));
        gen_stats.push_back(NAN);
        for (size_t i = 0; i < vif.size(); i++) gen_stats.push_back(vif[i]); 
    }

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

void LinearRegression::summary(bool detailled) const {
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
                  << std::setw(15) << stat.t_stat
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