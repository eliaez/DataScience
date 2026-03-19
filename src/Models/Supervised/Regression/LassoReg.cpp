#include <iomanip>
#include <iostream>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Stats/stats_reg.hpp"
#include "Validation/Validation.hpp"
#include "Models/Supervised/Regression/RidgeReg.hpp"
#include "Models/Supervised/Regression/LassoReg.hpp"

using namespace Utils;

namespace Reg {

std::pair<Dataframe, Dataframe> LassoRegression::fit_without_stats(const Dataframe& x, const Dataframe& y) {
    
    // Tests
    basic_verif(x);
    basic_verif(y);
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }

    size_t n = x.get_rows();
    size_t p = x.get_cols();

    // Center our data 
    auto [X_c, Y_c, x_mean] = center_data(x, y);
    
    X_c.change_layout_inplace();    // Need X_c row-major

    // Getting ptrs to elem of each col for coordinate descent
    std::vector<double> X_j_norm(p);
    std::vector<std::vector<const double*>> X_j(p);
    for (size_t j = 0; j < p; j++) {
        X_j[j] = X_c.getColumnPtrs(j);
        X_j_norm[j] = Lnorm(X_j[j], 2, 2);
    }

    // Soft Thresholding function
    auto soft_thres = [](
        double beta_tild,
        double omega
    ) {
        if (beta_tild > omega) {
            return beta_tild - omega;
        }
        else if (beta_tild < -omega) {
            return beta_tild + omega;
        }
        return 0.0;
    };

    // Coordinate descent 
    int idx = 0;
    bool keep_cond = true; 
    std::vector<double> v_beta_est(p, 0.0);
    Dataframe beta_est = {p, 1, false, v_beta_est};
    while (keep_cond && idx < 1000) {

        std::vector<double> beta_old = v_beta_est;
        for (size_t j = 0; j < p; j++) {

            // Core
            std::vector<double> r_j = add((Y_c - X_c * beta_est).get_data(), mult(X_j[j], v_beta_est[j]));
            double beta_tild = dot(X_j[j], r_j) / dot(X_j[j], X_j[j]);
            v_beta_est[j] = soft_thres(beta_tild, lambda_ * n / X_j_norm[j]);
            
            // Update
            beta_est = Dataframe(p, 1, false, v_beta_est);
        }

        // Testing convergence of beta_est
        keep_cond = false;
        std::vector<double> diff = sub(beta_old, v_beta_est);
        for (size_t i = 0; i < diff.size(); i++) {

            // Threshold 1e-4
            if (std::abs(diff[i]) > 1e-4) {
                keep_cond = true;
                break;
            } 
        }
        idx++;
    }

    // Results
    coeffs = beta_est.get_data();
    is_fitted = true;

    // Calculate and insert our intercept
    double intercept = Stats::mean(y.get_data()) - dot(x_mean, coeffs);
    coeffs.insert(coeffs.begin(), intercept);

    return {X_c, {}};
}

void LassoRegression::optimal_lambda(double start, double end, int nb, const Dataframe& x, const Dataframe& y) {
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
}

double LassoRegression::effective_df() const {
    double df = 0;
    for (size_t i = 1; i < coeffs.size(); i++) {
        if (std::abs(coeffs[i]) > 1e-10) df++;
    }
    return df;
}

std::unique_ptr<RegressionBase> LassoRegression::create(const std::vector<double>& params) {
    return std::make_unique<LassoRegression>(params[0]);
}

void LassoRegression::compute_stats(const Dataframe& x, Dataframe& x_c, Dataframe& XtXinv, const Dataframe& y) {
    
    RegressionBase::compute_stats_penalized(
        x, x_c, XtXinv, y,
        [this](Dataframe &/*a*/, Dataframe &/*b*/) {
            return effective_df();
    });
}

void LassoRegression::summary(bool detailled) const {
    RegressionBase::summary_penalized(lambda_, detailled);
}
}