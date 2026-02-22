#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Linalg/Linalg.hpp"
#include "Stats/stats_reg.hpp"
#include "Models/Regressions.hpp"

using namespace Utils;

namespace Reg {

std::string CoeffStats::significance() const {
    if (p_value < 0.001) return "***";
    if (p_value < 0.01)  return "** ";
    if (p_value < 0.05)  return "*  ";
    if (p_value < 0.10)  return ".  ";
    return "   ";
}

// ---------------------------------------RegBase------------------------------------------

void RegressionBase::basic_verif(const Dataframe& x) const {
    if (x.get_rows() == 0 || x.get_cols() == 0 || x.get_cols() < 1) {
        throw std::invalid_argument("Need non-empty input");
    }
}

std::vector<double> RegressionBase::predict(const Dataframe& x) const {
    basic_verif(x);

    if (!is_fitted) {
        throw std::runtime_error("Need to have trained your model");
    }

    size_t n = x.get_rows();
    size_t p = x.get_cols();
    std::vector<double> y_pred(n, 0.0);

    if (x.get_storage()) {
        #ifdef __AVX2__

            size_t vec_sizei = n - (n % NB_DB);
            size_t vec_sizej = p - (p % NB_DB);

            size_t i = 0;
            for (; i < vec_sizei; i+=NB_DB) {
                if (i + PREFETCH_DIST1 < vec_sizei) {
                    _mm_prefetch((const char*)&x.at((i+PREFETCH_DIST1)*p), _MM_HINT_T0);
                }
                __m256d sum0 = _mm256_setzero_pd();
                __m256d sum1 = _mm256_setzero_pd();
                __m256d sum2 = _mm256_setzero_pd();
                __m256d sum3 = _mm256_setzero_pd();

                size_t j = 0;
                for (; j < vec_sizej; j+=NB_DB) {
                    if (j + PREFETCH_DIST < vec_sizej) {
                        _mm_prefetch((const char*)&x.at(i*p + j + PREFETCH_DIST), _MM_HINT_T0);
                    }

                    __m256d vec_coeff = _mm256_loadu_pd(&coeffs[1 + j]);

                    __m256d vec0 = _mm256_loadu_pd(&x.at((i+0)*p+j));
                    __m256d vec1 = _mm256_loadu_pd(&x.at((i+1)*p+j));
                    __m256d vec2 = _mm256_loadu_pd(&x.at((i+2)*p+j));
                    __m256d vec3 = _mm256_loadu_pd(&x.at((i+3)*p+j));

                    sum0 = _mm256_fmadd_pd(vec_coeff, vec0, sum0);
                    sum1 = _mm256_fmadd_pd(vec_coeff, vec1, sum1);
                    sum2 = _mm256_fmadd_pd(vec_coeff, vec2, sum2);
                    sum3 = _mm256_fmadd_pd(vec_coeff, vec3, sum3);
                }

                // Horizontal reduction and add intercept
                y_pred[i+0] = coeffs[0] + horizontal_red(sum0);
                y_pred[i+1] = coeffs[0] + horizontal_red(sum1);
                y_pred[i+2] = coeffs[0] + horizontal_red(sum2);
                y_pred[i+3] = coeffs[0] + horizontal_red(sum3);

                // Scalar residual for j
                for (; j < p; j++) {
                    double c = coeffs[1 + j];
                    y_pred[i+0] += c * x.at((i+0)*p + j);
                    y_pred[i+1] += c * x.at((i+1)*p + j);
                    y_pred[i+2] += c * x.at((i+2)*p + j);
                    y_pred[i+3] += c * x.at((i+3)*p + j);
                }
            }

            // Scalar residual for i
            for (; i < n; i++) {
                
                double sum = 0.0;
                for (size_t j = 0; j < p; j++) {
                    sum += coeffs[1 + j] * x.at(i*p + j);
                }
                y_pred[i] += coeffs[0] + sum; // Add intercept
            }
        #else
            for (size_t i = 0; i < n; i++) {
                
                double sum = 0.0;
                for (size_t j = 0; j < p; j++) {
                    sum += coeffs[1 + j] * x.at(i*p + j);
                }
                y_pred[i] += coeffs[0] + sum; // Add intercept
            }
        #endif
    }
    else {
        #ifdef __AVX2__

            size_t i = 0;
            size_t vec_size = n - (n % NB_DB);

            for (size_t j = 0; j < p; j++) {

                i = 0;
                __m256d v_coeff = _mm256_set1_pd(coeffs[1 + j]);
                for (; i < vec_size; i+=NB_DB) {
                    if (i + PREFETCH_DIST < vec_size) {
                        _mm_prefetch((const char*)&x.at(j*n + i + PREFETCH_DIST), _MM_HINT_T0);
                    }

                    __m256d vec = _mm256_loadu_pd(&x.at(j*n + i));
                    __m256d vec_y = _mm256_loadu_pd(&y_pred[i]);
                    vec_y = _mm256_fmadd_pd(v_coeff, vec, vec_y);

                    _mm256_storeu_pd(&y_pred[i], vec_y);
                }

                // Scalar residual for i 
                double coeff = coeffs[1 + j];
                for (; i < n; i++) {
                    y_pred[i] += coeff * x.at(j*n + i);
                }
            }

            // Add intercept
            i = 0;
            __m256d v_intercept = _mm256_set1_pd(coeffs[0]);
            for (; i < vec_size; i+=NB_DB) {
                __m256d vec_y = _mm256_loadu_pd(&y_pred[i]);
                vec_y = _mm256_add_pd(vec_y, v_intercept);
                _mm256_storeu_pd(&y_pred[i], vec_y);
            }

            // Scalar residual for i
            for (; i < n; i++) {
                y_pred[i] += coeffs[0];
            }
        #else
            for (size_t j = 0; j < p; j++) {
                
                double coeff = coeffs[1 + j];
                for (size_t i = 0; i < n; i++) {
                    y_pred[i] += coeff * x.at(j*n + i);
                }
            }

            // Add intercept
            double intercept = coeffs[0];
            for (size_t i = 0; i < n; i++) {
                y_pred[i] += intercept;
            }
        #endif
    }

    return y_pred;
}

// ---------------------------------------LinearReg------------------------------------------

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
    if (!Omega_.get_data().empty()) {
        is_gls = true;
        basic_verif(Omega_);
        
        if (Omega_.get_cols() != n) {
            throw std::invalid_argument("Omega need to have n rows and cols with n = x.rows");
        }
        Om = Omega_.inv();
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
        std::vector<double> vif = Stats::OLS::VIF(x, Omega_);
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