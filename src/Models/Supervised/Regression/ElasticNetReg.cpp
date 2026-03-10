#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Stats/stats_reg.hpp"
#include "Validation/Validation.hpp"
#include "Models/Supervised/Regression/ElasticNetReg.hpp"

using namespace Utils;

namespace Reg {

std::pair<Dataframe, Dataframe> ElasticRegression::fit_without_stats(const Dataframe& x, const Dataframe& y) {
    
    // Tests
    basic_verif(x);
    basic_verif(y);
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }

    size_t p = x.get_cols();
    double lambda1 = alpha_ * l1_ratio_;
    double lambda2 = alpha_ * (1 - l1_ratio_);

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
    bool keep_cond = true; 
    std::vector<double> v_beta_est(p, 0.0);
    Dataframe beta_est = {p, 1, false, v_beta_est};
    while (keep_cond) {

        std::vector<double> beta_old = v_beta_est;
        for (size_t j = 0; j < p; j++) {

            // Core
            std::vector<double> r_j = add((Y_c - X_c * beta_est).get_data(), mult(X_j[j], v_beta_est[j]));
            double beta_tild = dot(X_j[j], r_j) / dot(X_j[j], X_j[j]);
            v_beta_est[j] = soft_thres(beta_tild, lambda1 / X_j_norm[j]) / (1 + lambda2 / X_j_norm[j]);
            
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
    }

    // Results
    coeffs = beta_est.get_data();
    is_fitted = true;

    // Calculate and insert our intercept
    double intercept = Stats::mean(y.get_data()) - dot(x_mean, coeffs);
    coeffs.insert(coeffs.begin(), intercept);

    return {X_c, {}};
}

double ElasticRegression::effective_df(Dataframe& X_c) const {

    basic_verif(X_c);
    size_t n = X_c.get_rows();
    size_t p = X_c.get_cols();
    double lambda2 = alpha_ * (1 - l1_ratio_);

    for (size_t i = (coeffs.size() - 1); i >= 1; i--) {
        if (std::abs(coeffs[i]) < 1e-10) {
            X_c.pop(i - 1);
            p--;
        }
    }

    // Lambda * Id Matrix
    std::vector<double> lambId(p*p, 0.0);
    for (size_t i = 0; i < p; i++) {
        lambId[i*p + i] = lambda2;
    }
    Dataframe LambId = {p, p, false, std::move(lambId)};

    // Transpose
    Dataframe X_t = ~X_c;
    X_t.change_layout_inplace();

    // Calculate XtXInv
    Dataframe XtXInv = (X_t*X_c + LambId).inv();
    X_c.change_layout_inplace(); // Need to be row_major for next operation

    // Calculate M matrix
    Dataframe M =  X_c * XtXInv;

    // Getting effective_df
    double df = 0.0;
    for (size_t i = 0; i < n; i++) {
        double hii = 0.0;
        for (size_t j = 0; j < p; j++)
            hii += M.at(i * p + j) * X_t.at(i * p + j);
        df += hii;
    }
    return df;
}

std::unique_ptr<RegressionBase> ElasticRegression::create(const std::vector<double>& params) {
    return std::make_unique<ElasticRegression>(params[0], params[1]);
}

void ElasticRegression::compute_stats(const Dataframe& x, Dataframe& x_c, Dataframe& XtXinv, const Dataframe& y) {
    
    RegressionBase::compute_stats_penalized(
        x, x_c, XtXinv, y,
        [this](Dataframe &a, Dataframe &/*b*/) {
            return effective_df(a);
    });
}

void ElasticRegression::summary(bool detailled) const {
    RegressionBase::summary_penalized(-1, detailled, alpha_, l1_ratio_);
}
}