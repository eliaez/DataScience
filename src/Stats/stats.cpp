#include "Stats/stats.hpp"
#include <stdexcept>

namespace Stats {

#ifdef __AVX2__
    double horizontal_red(__m256d& vec) {
        // hadd1 = [a+b, a+b, c+d, c+d] 
        __m256d hadd1 = _mm256_hadd_pd(vec, vec); 

        // sum128 = [a+b+c+d, ...]
        __m128d sum128 = _mm_add_pd(_mm256_castpd256_pd128(hadd1),  // [a+b, a+b]
                                    _mm256_extractf128_pd(hadd1, 1));  // [c+d, c+d]
        
        // Extract result
        return _mm_cvtsd_f64(sum128);
    }
#endif

double dot(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Need input of same length");
    }

    double res = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        res += x[i] * y[i];
    }
    return res;
}

double mean(const std::vector<double>& x) {
    if (x.empty()) {
        throw std::invalid_argument("Cannot calculate mean of empty vector");
    }

    double sum;
    size_t n = x.size();

    #ifdef __AVX2__
        size_t i = 0;
        size_t vec_size = n - (n % NB_DB);
        __m256d sum_vec = _mm256_set1_pd(0.0);

        for (; i < vec_size; i+=NB_DB) {
            // Pre-charged PREFETCH_DIST*8 bytes ahead
            if ( i + PREFETCH_DIST < vec_size) {
                _mm_prefetch((const char*)&x[i + PREFETCH_DIST], _MM_HINT_T0);
            }

            __m256d vec = _mm256_loadu_pd(&x[i]); 
            sum_vec = _mm256_add_pd(sum_vec, vec);
        }
        sum = horizontal_red(sum_vec);

        // Scalar residual
        for (; i < n; i++) {
            sum += x[i];
        }

    #else
        sum = 0.0;
        for (const double& val : x) {
            sum += val;
        }
    #endif

    return sum / n;
}

double var(const std::vector<double>& x) {
    if (x.empty() || x.size() < 2) {
        throw std::invalid_argument("Cannot calculate var with an empty vector or n < 2");
    }

    double sum;
    size_t n = x.size();
    double x_mean = mean(x);

    #ifdef __AVX2__
        size_t i = 0;
        size_t vec_size = n - (n % NB_DB);
        __m256d sum_vec = _mm256_set1_pd(0.0);
        __m256d mean_vec = _mm256_set1_pd(x_mean);

        for (; i < vec_size; i+=NB_DB) {
            // Pre-charged PREFETCH_DIST*8 bytes ahead
            if ( i + PREFETCH_DIST < vec_size) {
                _mm_prefetch((const char*)&x[i + PREFETCH_DIST], _MM_HINT_T0);
            }

            __m256d vec = _mm256_loadu_pd(&x[i]); 
            __m256d sub_vec = _mm256_sub_pd(vec, mean_vec); // (xi - x_mean)
            sum_vec = _mm256_fmadd_pd(sub_vec, sub_vec, sum_vec); // ... + (xi - x_mean)**2
        }
        sum = horizontal_red(sum_vec);

        // Scalar residual
        for (; i < n; i++) {
            sum += (x[i] - x_mean) * (x[i] - x_mean);
        }

    #else
        sum = 0.0;
        for (const double& val : x) {
            sum += (val - x_mean) * (val - x_mean); // ... + (xi - x_mean)**2
        }
    #endif

    return sum / (n-1);
}

double cov(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.empty() || x.size() < 2 || y.empty() || y.size() < 2 || x.size() != y.size()) {
        throw std::invalid_argument("Cannot calculate cov with an empty vector, n < 2 or x and y of different size");
    }

    double sum;
    size_t n = x.size();
    double x_mean = mean(x);
    double y_mean = mean(y);

    #ifdef __AVX2__
        size_t i = 0;
        size_t vec_size = n - (n % NB_DB);
        __m256d sum_vec = _mm256_set1_pd(0.0);
        __m256d x_mean_vec = _mm256_set1_pd(x_mean);
        __m256d y_mean_vec = _mm256_set1_pd(y_mean);

        for (; i < vec_size; i+=NB_DB) {
            // Pre-charged PREFETCH_DIST*8 bytes ahead
            if ( i + PREFETCH_DIST < vec_size) {
                _mm_prefetch((const char*)&x[i + PREFETCH_DIST], _MM_HINT_T0);
                _mm_prefetch((const char*)&y[i + PREFETCH_DIST], _MM_HINT_T0);
            }

            __m256d x_vec = _mm256_loadu_pd(&x[i]); 
            __m256d y_vec = _mm256_loadu_pd(&y[i]);

            __m256d x_sub_vec = _mm256_sub_pd(x_vec, x_mean_vec); // (xi - x_mean)
            __m256d y_sub_vec = _mm256_sub_pd(y_vec, y_mean_vec); // (yi - y_mean)

            sum_vec = _mm256_fmadd_pd(x_sub_vec, y_sub_vec, sum_vec); // ... + (xi - x_mean)*(yi - y_mean)
        }
        sum = horizontal_red(sum_vec);

        // Scalar residual
        for (; i < n; i++) {
            sum += (x[i] - x_mean) * (y[i] - y_mean);
        }

    #else
        sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            sum += (x[i] - x_mean) * (y[i] - y_mean); // ... + (xi - x_mean)*(yi - y_mean)
        }
    #endif

    return sum / (n-1);
}

double rsquared(const std::vector<double>& y, const std::vector<double>& y_pred) {
    if (y.empty() || y_pred.empty()) {
        throw std::invalid_argument("Cannot calculate rsquared with empty vector");
    }

    double SSres = 0.0;         // SSres - Sum of Squares of Residuals
    double SStot = 0.0;         // SStot - Total Sum of Squares
    double mean_y = mean(y);
    size_t n = y.size(); 
    for (size_t i = 0; i < n; i++) {
        SSres += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
        SStot += (y[i] - mean_y) * (y[i] - mean_y);
    }

    return 1 - SSres/SStot;
}

double radjusted(double r2, int n, int p) {
    if ((n - p - 1) != 0) {
        throw std::invalid_argument("Need more observations to avoid (n - p - 1) == 0");
    }
    
    return 1 - (1 - r2)*(n - 1)/(n - p - 1);
}

double mae(const std::vector<double>& y, const std::vector<double>& y_pred) {
    if (y.empty() || y_pred.empty()) {
        throw std::invalid_argument("Cannot calculate mae with empty vector");
    }

    double sum = 0.0;
    size_t n = y.size();
    for (size_t i = 0; i < n; i++) {
        sum += std::abs(y[i] - y_pred[i]);
    }
    
    return sum / n;
}

double mse(const std::vector<double>& y, const std::vector<double>& y_pred) {
    if (y.empty() || y_pred.empty()) {
        throw std::invalid_argument("Cannot calculate mse with empty vector");
    }

    double sum = 0.0;
    size_t n = y.size();
    for (size_t i = 0; i < n; i++) {
        sum += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
    }
    
    return sum / n; 
}

double fisher_test(double r2, int df1, int df2) {
    return (r2 * df2) / (df1 * (1 - r2));
}

double fisher_pvalue(double f, int df1, int df2) {
    
    // F dist with its degree of liberty
    boost::math::fisher_f dist(df1, df2);
    
    // P(F > f_obs) = 1 - CDF(f_obs)
    return 1.0 - boost::math::cdf(dist, f);
}

double normal_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

std::vector<double> student_pvalue(const std::vector<double>& t_stats) {

    size_t nb = t_stats.size();
    std::vector<double> pvalue;
    pvalue.reserve(nb);
    for (const double& t : t_stats) {
        pvalue.push_back(
            2 * (1 - normal_cdf(std::abs(t)))
        );
    }

    return pvalue;
}

}