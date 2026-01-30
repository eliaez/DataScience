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
            const size_t PREFETCH_DIST1 = Stats::PREFETCH_DIST1;

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
    
    std::vector<double> beta = {intercept, slope};
    size_t n = x.get_rows();
    size_t p = beta.size() - 1;
    
    // Degree of liberty
    int df1 = p;
    int df2 = n - df1 - 1;

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