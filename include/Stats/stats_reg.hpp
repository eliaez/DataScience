#pragma once

#include <vector>

// ---------------Forward Declaration----------------

class Dataframe;

// -----------------------Stats----------------------

namespace Stats {

// ------------------------------------General Stats--------------------------------------
    
    // Mean on a vector with Naive or AVX2
    double mean(const std::vector<double>& x);

    // Variance on a vector with Naive or AVX2
    double var(const std::vector<double>& x);

    // Covariance on vectors with Naive or AVX2
    double cov(const std::vector<double>& x, const std::vector<double>& y);

    // R2
    double rsquared(const std::vector<double>& y, const std::vector<double>& y_pred);

    // R2_adjusted for Regression with multiple features (n = nb of observations and p = nb of features)
    double radjusted(double r2, int n, int p);

    // MAE with n = nb of obsvervations
    double mae(const std::vector<double>& y, const std::vector<double>& y_pred);
    double mae(const std::vector<double>& residuals);

    // MSE with n = nb of obsvervations
    double mse(const std::vector<double>& y, const std::vector<double>& y_pred);
    double mse(const std::vector<double>& residuals);

    // RMSE with n = nb of obsvervations
    double rmse(double mse);

    // Cumulative distribution function of standard normal distribution
    double normal_cdf(double x);

    // Residuals
    std::vector<double> get_residuals(const std::vector<double>& y, const std::vector<double>& y_pred);

    // Stats on residuals, mean, stderr, abs max, Q1, Q2, Q3
    std::vector<double> residuals_stats(const std::vector<double>& residuals);

    // Durbin-Watson test return autocorrelation coefficient rho
    double durbin_watson_test(const std::vector<double>& residuals);

    // ---------------------------------------OLS Stats--------------------------------------
    namespace OLS {
        // Covariance Matrix for our Beta_est OLS with x col-major, 
        // cov_type : classical, HC3, HAC, cluster
        // And GLS with it's corresponding (Xt_OmegaInv_X)Inv 
        Dataframe cov_beta(
            const Dataframe& x_const, Dataframe& XtXinv, 
            const std::vector<double>& residuals, const std::string & cov_type = "classical",
            const std::vector<int>& cluster_ids = {}
        );

        // Fisher test with R2, df1 = p and df2 = n - p - 1, cov_type : classical, HC3, HAC, cluster and GLS
        double fisher_test(double r2, int df1, int df2, const std::vector<double>& beta_est,
            const Dataframe& cov_beta, const std::string& cov_type = "classical");

        // P Value for Fisher test with F value, df1 = p and df2 = n - p - 1
        double fisher_pvalue(double f, int df1, int df2);

        // Stderr of beta (take the diag of cov_beta)
        std::vector<double> stderr_b(const Dataframe& cov_beta);

        // Student p-value
        std::vector<double> student_pvalue(const std::vector<double>& t_stats);

        // Breusch-Pagan test return p-value (for Homoskedasticity)
        double breusch_pagan_test(const Dataframe& x, const std::vector<double>& residuals);

        // VIF (Variance Inflation Factor)
        std::vector<double> VIF(const Dataframe& x, const Dataframe& Omega = {});
    }

    namespace Regularized {
        double aic();
        double bic();
    }
}