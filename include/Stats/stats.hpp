#pragma once

#include <vector>
#include <cmath>
#include "Utils/ThreadPool.hpp"
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/chi_squared.hpp>

#ifdef __AVX2__
    #include <immintrin.h>
#endif

class Dataframe;

namespace Reg {
    class LinearRegression;
}

namespace Stats {
    
    #ifdef __AVX2__
        constexpr size_t NB_DB = 4; // AVX2 (256 bits) so 4 doubles
        constexpr size_t PREFETCH_DIST = 16; // Pre-fetch 16*64 bytes ahead for contigue memory only
        constexpr size_t PREFETCH_DIST1 = 4; // Pre-fetch 4*64 bytes ahead for Blocks algo

        // Horizontal Reduction
        double horizontal_red(__m256d& vec);
    #endif

    // Utils
    std::vector<size_t> rangeExcept(size_t max, size_t exclude);

    // Dot product 
    double dot(const std::vector<double>& x, const std::vector<double>& y);

    // Mean on a vector with Naive or AVX2
    double mean(const std::vector<double>& x);

    // Variance on a vector with Naive or AVX2
    double var(const std::vector<double>& x);

    // Covariance on vectors with Naive or AVX2
    double cov(const std::vector<double>& x, const std::vector<double>& y);

    // R2
    double rsquared(const std::vector<double>& y, const std::vector<double>& y_pred);

    // R2_adjusted for Polynomial Regression (n = nb of observations and p = nb of features)
    double radjusted(double r2, int n, int p);

    // MAE with n = nb of obsvervations
    double mae(const std::vector<double>& y, const std::vector<double>& y_pred);

    // MSE with n = nb of obsvervations
    double mse(const std::vector<double>& y, const std::vector<double>& y_pred);

    // RMSE with n = nb of obsvervations
    double rmse(double mse);

    // Fisher test with R2, df1 = p and df2 = n - p - 1
    double fisher_test(double r2, int df1, int df2);

    // P Value for Fisher test with F value, df1 = p and df2 = n - p - 1
    double fisher_pvalue(double f, int df1, int df2);

    // Cumulative distribution function of standard normal distribution
    double normal_cdf(double x);

    // Stderr of beta
    std::vector<double> stdderr_b(double mse, const std::vector<double>& XtXinv);

    // Student p-value
    std::vector<double> student_pvalue(const std::vector<double>& t_stats);

    // Residuals
    std::vector<double> get_residuals(const std::vector<double>& y, const std::vector<double>& y_pred);

    // Stats on residuals, mean, stdd, min, max, Q1, Q2, Q3
    std::vector<double> residuals_stats(const std::vector<double>& residuals);

    // Durbin-Watson test return autocorrelation coefficient
    double durbin_watson_test(const std::vector<double>& residuals);

    // Breusch-Pagan test return p-value (for heteroscedasticity)
    double breusch_pagan_test(const Dataframe& x, const std::vector<double>& residuals);

    // VIF (Variance Inflation Factor)
    std::vector<double> VIF(const Dataframe& x);
}