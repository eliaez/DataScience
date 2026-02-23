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
    
    auto [x_const, XtXInv] = fit_without_stats(x, y);
    compute_stats(x, x_const, XtXInv, y);
}

std::pair<Dataframe, Dataframe> RidgeRegression::fit_without_stats(const Dataframe& x, const Dataframe& y) {
    
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
}