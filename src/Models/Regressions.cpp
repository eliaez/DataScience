#include "Models/Regressions.hpp"
#include <stdexcept>

namespace Reg {

std::string CoeffStats::significance() const {
    if (p_value < 0.001) return "***";
    if (p_value < 0.01)  return "**";
    if (p_value < 0.05)  return "*";
    if (p_value < 0.10)  return ".";
    return "";
}

void LinearRegression::basic_verif(const Dataframe& x) const {
    if (x.get_rows() == 0 || x.get_cols() == 0 || x.get_cols() > 1) {
        throw std::invalid_argument("Need non-empty input");
    }
}

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

    compute_stats(x, inter, y);
}

std::vector<double> LinearRegression::predict(const Dataframe& x) const {
    basic_verif(x);

    if (!is_fitted) {
        throw std::runtime_error("Need to have trained your model");
    }

    const size_t n = x.get_rows();
    const size_t p = x.get_cols();
    std::vector<double> y_pred(n, 0.0);

    if (x.get_storage()) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; i < p; j++) {
                
            }
        }
        for (auto val : x.get_data()) {
            y_pred.push_back(val * slope + intercept);
        }
    }
    else {

    }

    return y_pred;
}

void LinearRegression::compute_stats(const Dataframe& x, const Dataframe& XtXinv, const Dataframe& y) {
    
    std::vector<double> beta = {intercept, slope};
    const size_t n = x.get_rows();
    const size_t p = beta.size() - 1;
    
    // Degree of liberty
    const int df1 = p;
    const int df2 = n - df1 - 1;

    // Predict 
    std::vector<double> y_pred = predict(x);

    // Calculate stats
    double r2 = Stats::rsquared(y.get_data(), y_pred);
    double mse = Stats::mse(y.get_data(), y_pred);
    double f_stat = Stats::fisher_test(r2, df1, df2);
    std::vector<double> stderr_beta = {std::sqrt(mse * XtXinv.get_data()[0]), std::sqrt(mse * XtXinv.get_data()[3])};
        
    // Add them to our vector of stats
    gen_stats.push_back(r2);
    gen_stats.push_back(mse);
    gen_stats.push_back(Stats::rmse(mse));
    gen_stats.push_back(Stats::mae(y.get_data(), y_pred));
    gen_stats.push_back(f_stat);
    gen_stats.push_back(Stats::fisher_pvalue(f_stat, df1, df2));

    // The t-distribution approaches the standard normal distribution for n > 30 
    if (n > 30) {
        std::vector<double> t_stats = {beta[0] / stderr_beta[0], beta[1] / stderr_beta[1]};
        std::vector<double> p_value = Stats::student_pvalue(t_stats);
    }
}


}