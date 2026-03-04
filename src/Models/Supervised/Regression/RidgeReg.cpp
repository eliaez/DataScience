#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Stats/stats_reg.hpp"
#include "Validation/Validation.hpp"
#include "Models/Supervised/Regression/RidgeReg.hpp"

using namespace Utils;

namespace Reg {

std::pair<Dataframe, Dataframe> RidgeRegression::fit_without_stats(const Dataframe& x, const Dataframe& y) {
    
    // Tests
    basic_verif(x);
    basic_verif(y);
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }

    size_t p = x.get_cols();

    // Center our data 
    auto [X_c, Y_c, x_mean] = center_data(x, y);

    // Lambda * Id Matrix
    std::vector<double> lambId(p*p, 0.0);
    for (size_t i = 0; i < p; i++) {
        lambId[i*p + i] = lambda_;
    }
    Dataframe LambId = {p, p, false, std::move(lambId)};

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

void RidgeRegression::optimal_lambda(double start, double end, int nb, const Dataframe& x, const Dataframe& y) {
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

double RidgeRegression::effective_df(Dataframe& X_c, Dataframe& XtXInv) const {

    basic_verif(X_c);
    basic_verif(XtXInv);
    size_t n = X_c.get_rows();

    // Need X_t col major 
    Dataframe X_t = ~X_c;           // Transpose change it to col-major
    if (!X_c.get_storage()) X_c.change_layout_inplace();        // Need to be row_major for next operation
    if (!XtXInv.get_storage()) XtXInv.change_layout_inplace();  // Need to be row_major for next operation

    // Calculate H matrix
    Dataframe H =  X_c * (XtXInv * X_t);  

    // Getting effectiv_df
    double df = 0.0;
    for (size_t i = 0; i < n; i++) {
        df += H.at(i * n + i);
    }
    return df;
}

std::unique_ptr<RegressionBase> RidgeRegression::create(const std::vector<double>& params) {
    return std::make_unique<RidgeRegression>(params[0]);
}

void RidgeRegression::compute_stats(const Dataframe& x, Dataframe& x_c, Dataframe& XtXinv, const Dataframe& y) {
    
    RegressionBase::compute_stats_penalized(
        x, x_c, XtXinv, y,
        [this](Dataframe &a, Dataframe &b) {
            return effective_df(a, b);
    });
}

void RidgeRegression::summary(bool detailled) const {
    RegressionBase::summary_penalized(lambda_, detailled);
}
}