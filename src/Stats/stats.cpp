#include <cmath>
#include <stdexcept>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Stats/stats.hpp"
#include "Utils/ThreadPool.hpp"
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/chi_squared.hpp>

using namespace Utils;

namespace Stats {

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

Dataframe cov_beta_OLS(const Dataframe& x_const, Dataframe& XtXinv, 
    const std::vector<double>& residuals, const std::string & cov_type,
    const std::vector<int>& cluster_ids) {

    size_t n = x_const.get_rows();
    size_t p = x_const.get_cols();
    std::vector<double> Meat(p*p, 0.0);
    
    #ifdef __AVX2__
        if (cov_type == "HC3") {
            
            // Calculate leverage vector
            std::vector<double> leverages(n);
            size_t vec_size = p - (p % NB_DB);
            for (size_t i = 0; i < n; i++) {
                
                double h = 0.0;
                for (size_t j = 0; j < p; j++) {
                    
                    size_t k = 0;
                    __m256d vec_h = _mm256_setzero_pd();
                    __m256d vec_temp = _mm256_set1_pd(x_const.at(j*n + i));
                    for (; k < vec_size; k+=NB_DB) {
                        if (k + PREFETCH_DIST < vec_size) {
                            _mm_prefetch((const char*)&XtXinv.at(j*p + k + PREFETCH_DIST), _MM_HINT_T0);
                        }

                        __m256d vec_temp1 = _mm256_set_pd(
                            x_const.at((k+3)*n + i),
                            x_const.at((k+2)*n + i),
                            x_const.at((k+1)*n + i),
                            x_const.at((k+0)*n + i)
                        );

                        __m256d vec_XtXinv = _mm256_loadu_pd(&XtXinv.at(j*p + k));
                        __m256d inter = _mm256_mul_pd(vec_XtXinv, vec_temp1);
                        vec_h = _mm256_fmadd_pd(vec_temp, inter, vec_h);
                    }
                    h += horizontal_red(vec_h);

                    // Scalar residual
                    double temp = x_const.at(j*n + i);
                    for (; k < p; k++) {
                        h += temp * XtXinv.at(j*p + k) * x_const.at(k*n + i);
                    }
                }
                leverages[i] = h;
            }

            // Meat = X' diag(weights) X
            for (size_t i = 0; i < n; i++) {

                double w = residuals[i] * residuals[i] / ((1 - leverages[i])*(1 - leverages[i]));
                for (size_t j = 0; j < p; j++) {

                    size_t k = 0;
                    double x_w = x_const.at(j*n + i) * w;
                    __m256d vec_x_w = _mm256_set1_pd(x_w);
                    for (; k < vec_size; k+=NB_DB) {
                        
                        __m256d vec_meat = _mm256_loadu_pd(&Meat[j*p + k]);
                        __m256d vec_temp1 = _mm256_set_pd(
                            x_const.at((k+3)*n + i),
                            x_const.at((k+2)*n + i),
                            x_const.at((k+1)*n + i),
                            x_const.at((k+0)*n + i)
                        );

                        vec_meat = _mm256_fmadd_pd(vec_x_w, vec_temp1, vec_meat);
                        _mm256_storeu_pd(&Meat[j*p + k], vec_meat);
                    }

                    // Scalar residual
                    for (; k < p; k++) {
                        Meat[j*p + k] += x_w * x_const.at(k*n + i);
                    }
                }
            }
        }
        else if (cov_type == "HAC") {

            // Newey-West lag
            size_t L = std::floor(4.0 * std::pow(n/100.0, 2.0/9.0));

            // Gamma_0
            size_t vec_size = p - (p % NB_DB);
            for (size_t i = 0; i < n; i++) {

                double residual = residuals[i];
                for (size_t j = 0; j < p; j++) {

                    size_t k = 0;
                    double x_residual = residual * x_const.at(j*n + i) * residual;
                    __m256d vec_residual = _mm256_set1_pd(x_residual);
                    for (; k < vec_size; k+=NB_DB) {
                        
                        __m256d vec_meat = _mm256_loadu_pd(&Meat[j*p + k]);
                        __m256d vec_temp1 = _mm256_set_pd(
                            x_const.at((k+3)*n + i),
                            x_const.at((k+2)*n + i),
                            x_const.at((k+1)*n + i),
                            x_const.at((k+0)*n + i)
                        );

                        vec_meat = _mm256_fmadd_pd(vec_residual, vec_temp1, vec_meat);
                        _mm256_storeu_pd(&Meat[j*p + k], vec_meat);
                    }

                    // Scalar residual
                    for (; k < p; k++) {
                        Meat[j*p + k] += x_residual * x_const.at(k*n + i);
                    }
                }
            }
            
            // Autocovariances
            for (size_t lag = 1; lag <= L; lag++) {
                
                double weight = 1.0 - static_cast<double>(lag) / (L + 1);
                std::vector<double> Gamma_lag(p*p, 0.0);
                for (size_t i = lag; i < n; i++) {

                    double res = residuals[i];
                    double res_lag = residuals[i-lag];
                    for (size_t j = 0; j < p; j++) {
                        
                        size_t k = 0;
                        double x_residual_lag = res * x_const.at(j*n + i) * res_lag;
                        __m256d vec_residual_lag = _mm256_set1_pd(x_residual_lag);
                        for (; k < vec_size; k+=NB_DB) {

                            __m256d vec_gamma = _mm256_loadu_pd(&Gamma_lag[j*p + k]);
                            __m256d vec_temp1 = _mm256_set_pd(
                                x_const.at((k+3)*n + i - lag),
                                x_const.at((k+2)*n + i - lag),
                                x_const.at((k+1)*n + i - lag),
                                x_const.at((k+0)*n + i - lag)
                            );

                            vec_gamma = _mm256_fmadd_pd(vec_residual_lag, vec_temp1, vec_gamma);
                            _mm256_storeu_pd(&Gamma_lag[j*p + k], vec_gamma);
                        }

                        // Scalar residual
                        for (; k < p; k++) {
                            Gamma_lag[j*p + k] +=  x_residual_lag * x_const.at(k*n + i - lag);
                        }
                    }
                }
                
                for (size_t j = 0; j < p; j++) {
                    
                    size_t k = 0;
                    __m256d vec_w = _mm256_set1_pd(weight);
                    for (; k < vec_size; k+=NB_DB) {

                        __m256d vec_meat = _mm256_loadu_pd(&Meat[j*p + k]);
                        __m256d vec_gamma = _mm256_loadu_pd(&Gamma_lag[j*p + k]);
                        __m256d vec_gamma1 = _mm256_set_pd(
                            Gamma_lag[(k+3)*p + j],
                            Gamma_lag[(k+2)*p + j],
                            Gamma_lag[(k+1)*p + j],
                            Gamma_lag[(k+0)*p + j]
                        );

                        vec_gamma = _mm256_add_pd(vec_gamma, vec_gamma1);
                        vec_meat = _mm256_fmadd_pd(vec_w, vec_gamma, vec_meat);
                        _mm256_storeu_pd(&Meat[j*p + k], vec_meat);
                    }

                    // Scalar residual
                    for (; k < p; k++) {
                        Meat[j*p + k] += weight * (Gamma_lag[j*p + k] + Gamma_lag[k*p + j]);
                    }
                }
            }
        }
        else if (cov_type == "cluster") {
                
            if (cluster_ids.empty()) {
                throw std::invalid_argument("Empty cluster_ids vector");
            }
            
            size_t min_val = *std::min_element(cluster_ids.begin(), cluster_ids.end());
            if (min_val != 0) {
                throw std::invalid_argument("Ids in cluster have to start from 0");
            }

            // Nb of clusters
            size_t G = *std::max_element(cluster_ids.begin(), cluster_ids.end()) + 1;

            // Group by Cluster
            std::vector<std::vector<size_t>> clusters(G);
            for (size_t i = 0; i < n; i++) {
                clusters[cluster_ids[i]].push_back(i);
            }

            size_t vec_size = p - (p % NB_DB);
            for (size_t c = 0; c < G; c++) {

                // Cluster score c : u_c = sum_{i in c} e_i * x_i
                std::vector<double> u_c(p, 0.0);
                for (size_t i : clusters[c]) {

                    size_t j = 0;
                    __m256d vec_residual = _mm256_set1_pd(residuals[i]);
                    for (; j < vec_size; j+=NB_DB) {

                        __m256d vec_uc = _mm256_loadu_pd(&u_c[j]);
                        __m256d vec_x = _mm256_set_pd(
                            x_const.at((j+3)*n + i),
                            x_const.at((j+2)*n + i),
                            x_const.at((j+1)*n + i),
                            x_const.at((j+0)*n + i)
                        );

                        vec_uc = _mm256_fmadd_pd(vec_residual, vec_x, vec_uc);
                        _mm256_storeu_pd(&u_c[j], vec_uc);
                    }

                    // Scalar residual
                    double residual = residuals[i];
                    for (; j < p; j++) {
                        u_c[j] += residual * x_const.at(j*n + i);
                    }
                }
                
                // Outer product u_c * u_c'
                for (size_t j = 0; j < p; j++) {

                    size_t k = 0;
                    __m256d vec_uc = _mm256_set1_pd(u_c[j]);
                    for (; k < vec_size; k+=NB_DB) {
                        if (k + PREFETCH_DIST < vec_size) {
                            _mm_prefetch((const char*)&u_c[k + PREFETCH_DIST], _MM_HINT_T0);
                        }

                        __m256d vec_meat = _mm256_loadu_pd(&Meat[j*p + k]);
                        __m256d uc_k = _mm256_loadu_pd(&u_c[k]);
                        vec_meat = _mm256_fmadd_pd(vec_uc, uc_k, vec_meat);

                        _mm256_storeu_pd(&Meat[j*p + k], vec_meat);
                    }

                    // Scalar residual
                    double temp = u_c[j];
                    for (; k < p; k++) {
                        Meat[j*p + k] += temp * u_c[k];
                    }
                }
            }
            
            // Adjustement
            double adj = static_cast<double>(G) / (G - 1) * (n - 1) / (n - p);
            Meat = mult(Meat, adj);
        }
        else {
            // Var(Beta) = sigma2 * XtXinv
            size_t i = 0;
            size_t vec_size = n - (n % NB_DB);
            __m256d vec_sigma2 = _mm256_setzero_pd();

            for (; i < vec_size; i+=NB_DB) {
                
                if (i + PREFETCH_DIST < vec_size) {
                    _mm_prefetch((const char*)&residuals[i + PREFETCH_DIST], _MM_HINT_T0);
                }

                __m256d vec_residuals = _mm256_loadu_pd(&residuals[i]);
                vec_sigma2 = _mm256_fmadd_pd(vec_residuals, vec_residuals, vec_sigma2);
            }
            double sigma2 = horizontal_red(vec_sigma2);

            // Scalar residual
            for (; i < n; i++) {
                sigma2 += residuals[i] * residuals[i];
            }
            sigma2 /= (n - p);

            // Id * sigma2
            i = 0;
            std::vector<double> sigma2_(p*p, 0.0);
            for (; i < p; i++) {
                sigma2_[i*p + i] = sigma2;
            }

            Dataframe df_sigma2 = {p, p, false, sigma2_};
            return XtXinv * df_sigma2;
        }
    #else
        if (cov_type == "HC3") {
            
            // Calculate leverage vector
            std::vector<double> leverages(n);
            for (size_t i = 0; i < n; i++) {
                
                double h = 0.0;
                for (size_t j = 0; j < p; j++) {
                    
                    double temp = x_const.at(j*n + i);
                    for (size_t k = 0; k < p; k++) {
                        h += temp * XtXinv.at(j*p + k) * x_const.at(k*n + i); // XtXinv row major
                    }
                }
                leverages[i] = h;
            }

            // Meat = X' diag(weights) X
            for (size_t i = 0; i < n; i++) {

                double w = residuals[i] * residuals[i] / ((1 - leverages[i])*(1 - leverages[i]));
                for (size_t j = 0; j < p; j++) {

                    double x_w = x_const.at(j*n + i) * w;
                    for (size_t k = 0; k < p; k++) {
                        Meat[j*p + k] += x_w * x_const.at(k*n + i);
                    }
                }
            }
        }
        else if (cov_type == "HAC") {

            // Newey-West lag
            size_t L = std::floor(4.0 * std::pow(n/100.0, 2.0/9.0));

            // Gamma_0
            for (size_t i = 0; i < n; i++) {

                double residual = residuals[i];
                for (size_t j = 0; j < p; j++) {

                    double x_residual = residual * x_const.at(j*n + i) * residual;
                    for (size_t k = 0; k < p; k++) {
                        Meat[j*p + k] += x_residual * x_const.at(k*n + i);
                    }
                }
            }
            
            // Autocovariances
            for (size_t lag = 1; lag <= L; lag++) {
                
                double weight = 1.0 - static_cast<double>(lag) / (L + 1);
                std::vector<double> Gamma_lag(p*p, 0.0);
                for (size_t i = lag; i < n; i++) {

                    double res = residuals[i];
                    double res_lag = residuals[i-lag];
                    for (size_t j = 0; j < p; j++) {

                        double x_residual_lag = res * x_const.at(j*n + i) * res_lag;
                        for (size_t k = 0; k < p; k++) {
                            Gamma_lag[j*p + k] +=  x_residual_lag * x_const.at(k*n + i - lag);
                        }
                    }
                }
                
                for (size_t j = 0; j < p; j++) {
                    for (size_t k = 0; k < p; k++) {
                        Meat[j*p + k] += weight * (Gamma_lag[j*p + k] + Gamma_lag[k*p + j]);
                    }
                }
            }
        }
        else if (cov_type == "cluster") {
                
            if (cluster_ids.empty()) {
                throw std::invalid_argument("Empty cluster_ids vector");
            }
            
            size_t min_val = *std::min_element(cluster_ids.begin(), cluster_ids.end());
            if (min_val != 0) {
                throw std::invalid_argument("Ids in cluster have to start from 0");
            }

            // Nb of clusters
            size_t G = *std::max_element(cluster_ids.begin(), cluster_ids.end()) + 1;

            // Group by Cluster
            std::vector<std::vector<size_t>> clusters(G);
            for (size_t i = 0; i < n; i++) {
                clusters[cluster_ids[i]].push_back(i);
            }

            for (size_t c = 0; c < G; c++) {

                // Cluster score c : u_c = sum_{i in c} e_i * x_i
                std::vector<double> u_c(p, 0.0);
                for (size_t i : clusters[c]) {

                    double residual = residuals[i];
                    for (size_t j = 0; j < p; j++) {
                        u_c[j] += residual * x_const.at(j*n + i);
                    }
                }
                
                // Outer product u_c * u_c'
                for (size_t j = 0; j < p; j++) {

                    double temp = u_c[j];
                    for (size_t k = 0; k < p; k++) {
                        Meat[j*p + k] += temp * u_c[k];
                    }
                }
            }
            
            // Adjustement
            double adj = static_cast<double>(G) / (G - 1) * (n - 1) / (n - p);
            Meat = mult(Meat, adj);
        }
        // By default classical
        else {
            // Var(Beta) = sigma2 * XtXinv
            double sigma2 = 0.0;
            for (size_t i = 0; i < n; i++) {
                sigma2 += residuals[i] * residuals[i];
            }
            sigma2 /= (n - p);

            // Id * sigma2
            std::vector<double> sigma2_vec(p*p, 0.0);
            for (size_t i = 0; i < p; i++) {
                sigma2_vec[i*p + i] = sigma2;
            }

            Dataframe df_sigma2 = {p, p, false, sigma2_vec};
            return XtXinv * df_sigma2;
        }
    #endif

    Dataframe df_Meat = {p, p, false, Meat};
    Dataframe part1 = XtXinv * df_Meat;
    part1.change_layout_inplace();
    XtXinv.change_layout_inplace();
    return part1 * XtXinv;
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
    if ((n - p - 1) == 0) {
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

double mae(const std::vector<double>& residuals) {
    if (residuals.empty()) {
        throw std::invalid_argument("Cannot calculate mae with empty vector");
    }

    double sum = 0.0;
    size_t n = residuals.size();
    for (size_t i = 0; i < n; i++) {
        sum += std::abs(residuals[i]);
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

double mse(const std::vector<double>& residuals) {
    if (residuals.empty()) {
        throw std::invalid_argument("Cannot calculate mse with empty vector");
    }

    double sum = 0.0;
    size_t n = residuals.size();
    for (size_t i = 0; i < n; i++) {
        sum += (residuals[i]) * (residuals[i]);
    }
    
    return sum / n; 
}

double rmse(double mse) { return std::sqrt(mse); }

double fisher_test(double r2, int df1, int df2, const std::vector<double>& beta_est,
    const Dataframe& cov_beta, const std::string& cov_type) {
    
    double f_stat;
    if (cov_type == "HC3" || cov_type == "HAC" || cov_type == "cluster" || cov_type == "GLS") {
        
        // Erase row and col 0
        Dataframe var_robust = cov_beta;
        var_robust.pop(0);
        var_robust.pop(0, true);
        var_robust = var_robust.inv();

        // Erase intercept
        std::vector<double> beta_no_const(beta_est.begin()+1, beta_est.end()); 
        size_t df1_size = static_cast<size_t>(df1);
        Dataframe df_beta_no_const = {1, df1_size, true, beta_no_const};

        double wald = dot((df_beta_no_const * var_robust).get_data(), beta_no_const);
        f_stat = wald / df1;
    }
    // By default classical
    else {
        f_stat = (r2 * df2) / (df1 * (1 - r2));
    }
    
    return f_stat;
}

double fisher_pvalue(double f, int df1, int df2) {
    
    // F dist with its degree of liberty
    boost::math::fisher_f dist(df1, df2);
    
    // P(F > f_obs) = 1 - CDF(f_obs)
    return 1.0 - boost::math::cdf(dist, f);
}

std::vector<double> stderr_b(const Dataframe& cov_beta) {
    
    size_t p = cov_beta.get_cols();
    std::vector<double> res(p);

    // Get diagonal of cov_beta
    for (size_t i = 0; i < p; i ++) {
        res[i] = std::sqrt(cov_beta.at(i*p + i));
    }
    return res;
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

std::vector<double> get_residuals(const std::vector<double>& y, const std::vector<double>& y_pred) {

    size_t n = y.size();
    std::vector<double> residuals(n, 0.0);
    for (size_t i = 0; i < n; i++) residuals[i] = y[i] - y_pred[i];
    return residuals;
}

std::vector<double> residuals_stats(const std::vector<double>& residuals) {

    double mean_res = mean(residuals);
    double stdd_res = std::sqrt(var(residuals));
    double max_abs = std::abs(residuals[0]);

    for (const double& r : residuals) {
        max_abs = std::max(max_abs, std::abs(r));
    }

    size_t n = residuals.size();
    auto sorted_residuals = residuals;
    std::sort(sorted_residuals.begin(), sorted_residuals.end());

    auto calc_quantile = [&](double q) {
        double pos = q * (n - 1);
        int idx = static_cast<int>(pos);
        double frac = pos - idx;
        return (idx + 1 < n) ? 
            sorted_residuals[idx] + frac * (sorted_residuals[idx + 1] - sorted_residuals[idx]) : sorted_residuals[idx];
    };

    double Q1 = calc_quantile(0.25);
    double Q2 = calc_quantile(0.5);
    double Q3 = calc_quantile(0.75);

    std::vector<double> res_stats = {mean_res, stdd_res, max_abs, Q1, Q2, Q3};
    return res_stats;
}

double durbin_watson_test(const std::vector<double>& residuals) {

    double sum1 = 0.0;
    double sum2 = 0.0;
    size_t n = residuals.size();
    for (size_t i = 0; i < n; i++) {
        sum2 += residuals[i] * residuals[i]; 
    }
    for (size_t i = 1; i < n; i++) {
        sum1 += (residuals[i] - residuals[i-1]) * (residuals[i] - residuals[i-1]); 
    }

    return 1 - (sum1/sum2)/2;
}

double breusch_pagan_test(const Dataframe& x, const std::vector<double>& residuals) {
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    
    // Squared residuals 
    std::vector<double> squared_res = residuals;
    for (size_t i = 0; i < n; i++) squared_res[i] = residuals[i] * residuals[i];

    // Use our functions in Reg
    Dataframe target = {n, 1, false, squared_res};
    Reg::LinearRegression New_reg;
    New_reg.fit_without_stats(x, target);
    
    auto y_pred = New_reg.predict(x);
    double r2 = rsquared(target.get_data(), y_pred);

    double LM = n * r2;
    boost::math::chi_squared_distribution<double> chi2(p);
    
    // p-value = P(X > LM) = 1 - CDF(LM)
    double p_value = 1.0 - boost::math::cdf(chi2, LM);
    return p_value;
}

std::vector<double> VIF(const Dataframe& x, const Dataframe& Omega) {
    
    size_t p = x.get_cols();
    std::vector<double> vif(p, 0.0);
    
    // For each predictor variable X_j, we measure how predictable it is by the other variables.
    for (size_t i = 0; i < p; i++) {
        
        // Target and X for this reg 
        Dataframe target = x[i];
        Dataframe x_bis = x;
        x_bis.pop(i);

        // Use our functions in Reg
        Reg::LinearRegression New_reg;

        // In the case of GLS LinearRegression
        std::vector<double> y_pred;
        if (!Omega.get_data().empty()) {
            Dataframe Om = Omega;
            Om.pop(i);
            Om.pop(i, true);
            New_reg.fit_gls_without_stats(x_bis, target, Om);
            y_pred = New_reg.predict(x_bis);
        }
        // In the case of OLS LinearRegression
        else {
            New_reg.fit_without_stats(x_bis, target);
            y_pred = New_reg.predict(x_bis);
        }

        // VIF
        double r2 = rsquared(target.get_data(), y_pred);
        if (r2 >= 0.9999) {
            vif[i] = INFINITY;
        } else {
            vif[i] = 1.0 / (1.0 - r2);
        }
    }
    return vif;
}

}