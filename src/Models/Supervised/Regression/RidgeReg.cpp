#include <iomanip>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Linalg/Linalg.hpp"
#include "Stats/stats_reg.hpp"
#include "Models/Supervised/Regression/RidgeReg.hpp"

using namespace Utils;

namespace Reg {

void RidgeRegression::fit(const Dataframe& x, const Dataframe& y) {
    
    auto [x_c, XtXInv] = fit_without_stats(x, y);
    compute_stats(x, x_c, XtXInv, y);
}

std::pair<Dataframe, Dataframe> RidgeRegression::fit_without_stats(const Dataframe& x, const Dataframe& y) {
    
    // Tests
    basic_verif(x);
    basic_verif(y);
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }

    size_t n = x.get_rows();

    // Center our data 
    auto [X_c, Y_c, x_mean] = center_data(x, y);

    // Lambda * Id Matrix
    std::vector<double> lambId(n*n, 0.0);
    for (size_t i = 0; i < n; i++) {
        lambId[i*n + i] = lambda_;
    }
    Dataframe LambId = {n, n, false, std::move(lambId)};

    // Need X_t row-major (for mult ops)
    Dataframe X_t = ~X_c;  // Transpose change it to col-major
    X_t.change_layout_inplace();

    // Calculate Beta (our estimator) for Ridge Regression
    Dataframe XtXInv = (X_t*X_c + LambId).inv();
    XtXInv.change_layout_inplace();
    Dataframe beta_est =  XtXInv * (X_t * Y_c);  

    // Results
    coeffs = beta_est.get_data();
    is_fitted = true;

    // Calculate and insert our intercept
    double intercept = Stats::mean(y.get_data()) - dot(x_mean, coeffs);
    coeffs.insert(coeffs.begin(), intercept);

    return {X_c, XtXInv};
}

std::vector<double> RidgeRegression::lambda_path(double start, double end, int nb) const {
    std::vector<double> path(nb);
    double log_min = log(start);
    double log_max = log(end);
    double step = (log_max - log_min) / (nb - 1);

    for (int i = 0; i < nb; i++) {
        path[i] = exp(log_min + i * step);
    }
    return path;
}

double RidgeRegression::effective_df(const Dataframe& x) const {

    // Convert to use Eigen function (see doc for more details)
    Eigen::Map<const Eigen::MatrixXd> X(
        x.get_db(), 
        static_cast<Eigen::Index>(x.get_rows()), 
        static_cast<Eigen::Index>(x.get_cols())
    );

    // SVD
    Eigen::BDCSVD<Eigen::MatrixXd> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Singular Values desc
    Eigen::VectorXd singular_values = svd.singularValues();
    std::vector<double> sv(singular_values.data(), singular_values.data() + singular_values.size());

    // Calculate df
    double df = 0.0;
    for (size_t i = 0; i < sv.size(); i++) {
        df += sv[i]*sv[i] / (sv[i]*sv[i] + lambda_);
    }
    return df;
}

std::unique_ptr<RegressionBase> create(const std::vector<double>& params) {
    return std::make_unique<RidgeRegression>(params[0]);
}

void RidgeRegression::compute_stats(const Dataframe& x, const Dataframe& x_c, Dataframe& XtXinv, const Dataframe& y) {
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();

    // Predict 
    std::vector<double> y_pred = predict(x);

    // -------------------------------------Calculate stats----------------------------------------
    double r2 = Stats::rsquared(y.get_data(), y_pred);
    double mse = Stats::mse(y.get_data(), y_pred);
    std::vector<double> residuals = Stats::get_residuals(y.get_data(), y_pred);
    double df = effective_df(x_c);
    double loglikehood = Stats::logLikehood(y.get_data(), y_pred);


    // Add them to our vector of stats
    gen_stats.push_back(r2);

    if (p > 1) gen_stats.push_back(Stats::radjusted(r2, n, p));
    else gen_stats.push_back(-1.0);

    gen_stats.push_back(mse);
    gen_stats.push_back(Stats::rmse(mse));
    gen_stats.push_back(Stats::mae(y.get_data(), y_pred));
    gen_stats.push_back(Stats::Regularized::AIC(df, loglikehood));
    gen_stats.push_back(Stats::Regularized::BIC(df, loglikehood, n));
    gen_stats.push_back(Stats::durbin_watson_test(residuals));

    std::vector<double> resid_stats = Stats::residuals_stats(residuals);
    for (size_t i = 0; i < resid_stats.size(); i++) gen_stats.push_back(resid_stats[i]); 

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
        c = {
            headers[i],
            coeffs[i],
            NAN,
            NAN,
            NAN
        };
        coeff_stats.push_back(c);
    }
}

void RidgeRegression::summary(bool detailled) const {
    std::cout << "\n=== REGRESSION SUMMARY ===\n\n";
    
    std::cout << "Choosen lambda: " << lambda_ << "\n"
              << "R2 = " << gen_stats[0] << "\n";

    if (gen_stats[1] != -1.0) std::cout << "Adjusted R2 = " << gen_stats[1] << "\n";
    
    std::cout << "MSE = " << gen_stats[2] << "\n"
              << "RMSE = " << gen_stats[3] << "\n"
              << "MAE = " << gen_stats[4] << "\n"
              << "AIC = " << gen_stats[5] << "\n"
              << "BIC = " << gen_stats[6] << "\n\n";
    
    std::cout << std::left << std::setw(15) << "Coefficient"
              << std::right << std::setw(15) << "Beta"
              << "  \n"
              << std::string(35, '-') << "\n";

    size_t i = 0;
    for (const auto& stat : coeff_stats) {
        std::cout << std::left << std::setw(15) << stat.name
                  << std::right << std::fixed << std::setprecision(4)
                  << std::setw(15) << stat.beta
                  << "\n";
        i++;
    }
    std::cout << std::endl;

    if (detailled) {
        std::cout << "Additional stats:\n";
        std::cout << "Durbin-Watson - rho-value = " << gen_stats[7] << "\n\n";

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
                << std::setw(15) << gen_stats[8]
                << std::setw(15) << gen_stats[9]
                << std::setw(15) << gen_stats[10]
                << std::setw(15) << gen_stats[11]
                << std::setw(15) << gen_stats[12]
                << std::setw(15) << gen_stats[13] << "\n" << std::endl;
    }
}
}