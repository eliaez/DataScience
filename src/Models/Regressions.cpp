#include "Models/Regressions.hpp"
#include <stdexcept>

namespace Reg {

void LinearRegression::fit(const Dataframe& x, const Dataframe& y) {
    basic_verif(x);
    basic_verif(y);

    // Copy our data 
    std::vector<double> x_v = x.get_data();
    
    // Insert an unit col to get intercept value
    size_t n = x_v.size();
    bool is_row_major = x.get_storage();
    if (is_row_major) {
        for (size_t i = 0; i < n; i++) {
            x_v.insert(x_v.begin() + i*2, 1.0);
        }
    }
    else {
        for (size_t i = 0; i < n; i++) {
            x_v.insert(x_v.begin(), 1.0);
        }
    }

    // Need X col-major (for mult ops)
    Dataframe X = {n, 2, is_row_major, std::move(x_v)};
    if (is_row_major) X.change_layout_inplace();

    // Need X_t row-major (for mult ops)
    Dataframe X_t = ~X;  // Transpose change it to col-major
    X_t.change_layout_inplace();
    
    // Calculate Beta (our estimator)
    Dataframe inter = (X_t*X).inv();
    inter.change_layout_inplace();    
    Dataframe beta_est =  inter * (X_t * y);  

    // Results
    intercept = beta_est.get_data()[0];
    slope = beta_est.get_data()[1];
    is_fitted = true;

    // Calculate Stats
    std::vector<double> y_pred = predict(x);
    double mse = Stats::mse(y.get_data(), y_pred);
    std::vector<double> stderr_beta = {std::sqrt(mse * inter.get_data()[0]), std::sqrt(mse * inter.get_data()[3])};
    std::vector<double> beta = {intercept, slope};
    
    if (n > 30) {
        std::vector<double> t_stat = {beta[0] / stderr_beta[0], beta[1] / stderr_beta[1]};
    }
    
    v_stats.push_back(Stats::rsquared(y.get_data(), y_pred));
    v_stats.push_back(mse);
    v_stats.push_back(Stats::rmse(mse));
    v_stats.push_back(Stats::mae(y.get_data(), y_pred));

}

std::vector<double> LinearRegression::predict(const Dataframe& x) const {
    basic_verif(x);

    if (!is_fitted) {
        throw std::runtime_error("Need to have trained your model");
    }

    std::vector<double> y_pred;
    y_pred.reserve(x.get_rows());
    for (auto val : x.get_data()) {
        y_pred.push_back(val * slope + intercept);
    }
    return y_pred;
}

void LinearRegression::basic_verif(const Dataframe& x) const {
    if (x.get_rows() == 0 || x.get_cols() == 0 || x.get_cols() > 1) {
        throw std::invalid_argument("Need non-empty input");
    }
}


}