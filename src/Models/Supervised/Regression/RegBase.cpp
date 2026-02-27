#include <iomanip>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Linalg/Linalg.hpp"
#include "Stats/stats_reg.hpp"
#include "Models/Supervised/Regression/RegBase.hpp"

using namespace Utils;

namespace Reg {

std::string CoeffStats::significance() const {
    if (p_value < 0.001) return "***";
    if (p_value < 0.01)  return "** ";
    if (p_value < 0.05)  return "*  ";
    if (p_value < 0.10)  return ".  ";
    return "   ";
}

void RegressionBase::basic_verif(const Dataframe& x) const {
    if (x.get_rows() == 0 || x.get_cols() == 0 || x.get_cols() < 1) {
        throw std::invalid_argument("Need non-empty input");
    }
}

std::unique_ptr<RegressionBase> RegressionBase::create(const std::vector<double>& params) {
    throw std::logic_error("GridSearch not supported for this model");
}

std::tuple<Dataframe, Dataframe, std::vector<double>> RegressionBase::center_data(const Dataframe& x, const Dataframe& y) const {
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();

    // Starting by centering Y
    std::vector<double> y_c = y.get_data();
    double y_mean = Stats::mean(y.get_data());
    for (size_t i = 0; i < n; i++) {
        y_c[i] -= y_mean;
    }

    // Centering X
    std::vector<double> x_mean(p);
    std::vector<double> x_c = x.get_data();
    for (size_t i = 0; i < p; i++) {
        
        // Pointer to the start of column i
        double* col_start = x_c.data() + i * n;
        
        // Mean
        x_mean[i] = std::accumulate(col_start, col_start + n, 0.0) / n;
        
        // Center
        for (size_t j = 0; j < n; j++) {
            col_start[j] -= x_mean[i];
        }
    }

    Dataframe X_c = {n, p, false, std::move(x_c)};
    Dataframe Y_c = {n, 1, false, std::move(y_c)};

    return {X_c, Y_c, x_mean};
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
}