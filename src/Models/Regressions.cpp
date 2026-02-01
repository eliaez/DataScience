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
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    bool is_row_major = x.get_storage();
    if (is_row_major) {
        for (size_t i = 0; i < n; i++) {
            x_v.insert(x_v.begin() + i*(p+1), 1.0);
        }
    }
    else {
        for (size_t i = 0; i < n; i++) {
            x_v.insert(x_v.begin(), 1.0);
        }
    }

    // Need X col-major (for mult ops)
    Dataframe X = {n, p+1, is_row_major, std::move(x_v)};
    if (is_row_major) X.change_layout_inplace();

    // Need X_t row-major (for mult ops)
    Dataframe X_t = ~X;  // Transpose change it to col-major
    X_t.change_layout_inplace();
    
    // Calculate Beta (our estimator)
    Dataframe inter = (X_t*X).inv();
    inter.change_layout_inplace();    
    Dataframe beta_est =  inter * (X_t * y);  

    // Results
    coeffs = beta_est.get_data();
    is_fitted = true;

    compute_stats(x, inter, y);
}

std::vector<double> LinearRegression::predict(const Dataframe& x) const {
    basic_verif(x);

    if (!is_fitted) {
        throw std::runtime_error("Need to have trained your model");
    }

    size_t n = x.get_rows();
    size_t p = x.get_cols();
    std::vector<double> y_pred(n, 0.0);

    if (x.get_storage()) {
        #ifdef __AVX2__
            const size_t NB_DB = Stats::NB_DB;
            const size_t PREFETCH_DIST = Stats::PREFETCH_DIST;
            const size_t PREFETCH_DIST1 = Stats::PREFETCH_DIST1;

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
                y_pred[i+0] = coeffs[0] + Stats::horizontal_red(sum0);
                y_pred[i+1] = coeffs[0] + Stats::horizontal_red(sum1);
                y_pred[i+2] = coeffs[0] + Stats::horizontal_red(sum2);
                y_pred[i+3] = coeffs[0] + Stats::horizontal_red(sum3);

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
            const size_t NB_DB = Stats::NB_DB;
            const size_t PREFETCH_DIST = Stats::PREFETCH_DIST;

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

void LinearRegression::compute_stats(const Dataframe& x, const Dataframe& XtXinv, const Dataframe& y) {
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    
    // Degree of liberty
    int df1 = p;
    int df2 = n - df1 - 1;

    // Predict 
    std::vector<double> y_pred = predict(x);

    // Calculate stats
    double r2 = Stats::rsquared(y.get_data(), y_pred);
    double mse = Stats::mse(y.get_data(), y_pred);
    double f_stat = Stats::fisher_test(r2, df1, df2);
    std::vector<double> stderr_beta = Stats::stderr_b(mse, XtXinv.get_data());
        
    // Add them to our vector of stats
    gen_stats.push_back(r2);

    if (p > 1) gen_stats.push_back(Stats::radjusted(r2, n, p));
    else gen_stats.push_back(-1.0);

    gen_stats.push_back(mse);
    gen_stats.push_back(Stats::rmse(mse));
    gen_stats.push_back(Stats::mae(y.get_data(), y_pred));
    gen_stats.push_back(f_stat);
    gen_stats.push_back(Stats::fisher_pvalue(f_stat, df1, df2));

    // The t-distribution approaches the standard normal distribution for n > 30 
    std::vector<double> p_value;
    std::vector<double> t_stats(p+1, 0.0);
    if (n > 30) {
        for (size_t i = 0; i < p+1; i++) t_stats[i] = coeffs[i] / stderr_beta[i];
        p_value = Stats::student_pvalue(t_stats);
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
                stderr_beta[i],
                t_stats[i],
                p_value[i]
            };
        }
        else {
            c = {
                headers[i],
                coeffs[i],
                stderr_beta[i],
                0.0,
                0.0
            };
        }
        coeff_stats.push_back(c);
    }
}

void LinearRegression::summary() const {
    std::cout << "\n=== REGRESSION SUMMARY ===\n\n";
    
    std::cout << "R² = " << gen_stats[0] << "\n";
    if (gen_stats[1] != -1.0) std::cout << "Adjusted R² = " << gen_stats[1] << "\n";
    std::cout << "MSE = " << gen_stats[2] << "\n";
    std::cout << "RMSE = " << gen_stats[3] << "\n";
    std::cout << "MAE = " << gen_stats[4] << "\n";
    std::cout << "Fisher - F = " << gen_stats[5] << "\n";
    std::cout << "Fisher - p-value = " << gen_stats[6] << "\n\n";
    
    std::cout << std::left << std::setw(15) << "Coefficient"
              << std::right << std::setw(12) << "Beta"
              << std::setw(12) << "Stderr"
              << std::setw(10) << "t-stat"
              << std::setw(10) << "p-value"
              << "  \n";
    std::cout << std::string(60, '-') << "\n";
    
    for (const auto& stat : coeff_stats) {
        std::cout << std::left << std::setw(15) << stat.name
                  << std::right << std::fixed << std::setprecision(4)
                  << std::setw(12) << stat.beta
                  << std::setw(12) << stat.stderr_beta
                  << std::setw(10) << stat.t_stat
                  << std::setw(10) << stat.p_value
                  << "  " << stat.significance() << "\n";
    }
    
    std::cout << "\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1\n";
}
}