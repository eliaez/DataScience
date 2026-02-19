#include "Data/Data.hpp"
#include "LinalgAVX2.hpp"
#include "Utils/Utils.hpp"

using namespace Utils;

namespace Linalg::AVX2 {
#ifdef __AVX2__

std::tuple<int, std::vector<double>, std::vector<double>> LU_decomposition(const std::vector<double>& v1, size_t n) {

    int nb_swaps = 0;
    std::vector<double> LU = v1;

    // Permutation matrix is Id at first
    std::vector<double> swaps(n*n, 0.0);
    for (size_t i = 0; i < n; i++) {
        swaps[i*n + i] = 1;
    }

    for (size_t k = 0; k < n-1; k++) {

        // Partial pivot (get most important pivot and permutate lines)
        auto [max, idx] = std::tuple{LU[k*n + k], k};
        for (size_t i = k+1; i < n; i++) {

            double val = std::abs(LU[k*n + i]);
            if (max < val) {
                max =  val;
                idx = i;
            }
        }

        // Permutation of rows
        if (k != static_cast<size_t>(idx)) {
            for (size_t j = 0; j < n; j++) {
                std::swap(LU[j*n + k], LU[j*n + idx]);
                std::swap(swaps[j*n + k], swaps[j*n + idx]);
            }
            nb_swaps++;
        }

        // Pivot
        double p = LU[k*n + k];
        if (std::abs(p) < 1e-14) {
            // det = 0
            throw std::runtime_error("Singular Matrix <=> Det = 0");
        }

        // LU decomposition algorithm by blocks AVX2
        size_t vec_size = n - ((n - k - 1) % NB_DB);
        
        for (size_t i = k+1; i < n; i++) {

            double Lvalue = LU[k*n + i] / p;
            LU[k*n + i] = Lvalue;
            
            // Broadcast our L value in vector
            __m256d vec_Lvalue = _mm256_set1_pd(Lvalue);
            
            // Update value in other cols 
            // Block version
            size_t j = k+1;
            for (; j < vec_size; j+=NB_DB) {
                        
                __m256d vec_pivot = _mm256_set_pd(
                    LU[(j+3)*n + k],
                    LU[(j+2)*n + k],
                    LU[(j+1)*n + k],
                    LU[j*n + k]
                );
                
                __m256d vec_LU = _mm256_set_pd(
                    LU[(j+3)*n + i],
                    LU[(j+2)*n + i],
                    LU[(j+1)*n + i],
                    LU[j*n + i]
                );
                
                // (a*b) - c
                vec_LU = _mm256_fnmadd_pd(vec_Lvalue, vec_pivot, vec_LU);

                alignas(32) double temp[4];
                _mm256_store_pd(temp, vec_LU);
                LU[j*n + i] = temp[0];
                LU[(j+1)*n + i] = temp[1];
                LU[(j+2)*n + i] = temp[2];
                LU[(j+3)*n + i] = temp[3];
            }

            // Scalar residual
            for (; j < n; j++) {
                LU[j*n + i] -= Lvalue * LU[j*n + k];
            }
        }
    }
    return std::make_tuple(nb_swaps, std::move(swaps), std::move(LU));
}

std::vector<double> solveLU_inplace(const std::vector<double>& perm, const std::vector<double>& LU, size_t n) {

    std::vector<double> y(n*n, 0.0);

    // Store the Diag
    std::vector<double> diag_U(n);
    for (size_t i = 0; i < n; i++) {
        diag_U[i] = LU[i * n + i];
    }

    size_t k = 0;
    size_t vec_size = n - (n % NB_DB);

    // By Blocks with AVX2
    for (; k < vec_size; k+=NB_DB) {

        // Solving Ly = perm (No division because diag = 1)
        // Forward substitution 
        for (size_t i = 0; i < n; i++) {

            if (k + PREFETCH_DIST1 < vec_size) {
                for (size_t p = 0; p < NB_DB; p++) {
                    _mm_prefetch((const char*)&perm[(k+p+PREFETCH_DIST1)*n + i], _MM_HINT_T0);
                }        
            }

            __m256d sum = _mm256_set_pd(
                perm[(k+3)*n + i],
                perm[(k+2)*n + i],
                perm[(k+1)*n + i],
                perm[(k+0)*n + i]
            );
            
            for (size_t j = 0; j < i; j++) {

                if (j + PREFETCH_DIST1 < i) {
                    for (size_t p = 0; p < NB_DB; p++) {
                        _mm_prefetch((const char*)&y[(k+p)*n + j + PREFETCH_DIST1], _MM_HINT_T0);
                    }
                }

                __m256d LU_val = _mm256_set1_pd(LU[j*n + i]);
                
                __m256d y_vals = _mm256_set_pd(
                    y[(k+3)*n + j],
                    y[(k+2)*n + j],
                    y[(k+1)*n + j],
                    y[(k+0)*n + j]
                );
                sum = _mm256_fnmadd_pd(LU_val, y_vals, sum);
            }
            
            double result[4];
            _mm256_storeu_pd(result, sum);

            y[(k+0)*n + i] = result[0];
            y[(k+1)*n + i] = result[1];
            y[(k+2)*n + i] = result[2];
            y[(k+3)*n + i] = result[3];
        }

        // Solving Ux = y 
        // Backward substitution
        for (int i = static_cast<int>(n)-1; i >= 0; i--) {

            __m256d sum = _mm256_set_pd(
                y[(k+3)*n + i],
                y[(k+2)*n + i],
                y[(k+1)*n + i],
                y[(k+0)*n + i]
            );
            
            for (size_t j = i+1; j < n; j++) {
                
                __m256d LU_vec = _mm256_set1_pd(LU[j*n + i]);

                __m256d y_vals = _mm256_set_pd(
                    y[(k+3)*n + j],
                    y[(k+2)*n + j],
                    y[(k+1)*n + j],
                    y[(k+0)*n + j]
                );
                
                sum = _mm256_fnmadd_pd(LU_vec, y_vals, sum);
            }
            
            double result[4];
            _mm256_storeu_pd(result, sum);
            
            y[(k+0)*n + i] = result[0];
            y[(k+1)*n + i] = result[1];
            y[(k+2)*n + i] = result[2];
            y[(k+3)*n + i] = result[3];

            for(size_t p = 0; p < NB_DB; p++) {
                if (std::abs(y[(k+p)*n + i]) < 1e-14) y[(k+p)*n + i] = 0;
                else y[(k+p)*n + i] /= diag_U[i];
            }
        }
    }

    // Scalar Residual for k 
    for (; k < n; k++) {

        // Solving Ly = perm (No division because diag = 1)
        // Forward substitution
        for (size_t i = 0; i < n; i++) {
            
            double sum = 0.0;
            sum = perm[k*n + i];
            for (size_t j = 0; j < i; j++) {
                sum -= LU[j*n + i] * y[k*n + j];
            }
            y[k*n + i] = sum;
        }

        // Solving Ux = y 
        // Backward substitution
        for (int i = static_cast<int>(n)-1; i >= 0; i--) {

            double sum = y[k*n + i];
            for (size_t j = i+1; j < n; j++) {
                sum -= LU[j*n + i] * y[k*n + j];
            }
            y[k*n + i] = sum;
            if (std::abs(y[k*n + i]) < 1e-14) y[k*n + i] = 0;
            else y[k*n + i] /= diag_U[i];
            
        }
    }
    return y;
}

std::vector<double> sum(const std::vector<double>& v1, const std::vector<double>& v2, 
    size_t m, size_t n, char op) { 

    // New data
    std::vector<double> new_data(m * n);
    
    size_t i = 0;
    size_t vec_size = m*n - ((m*n) % NB_DB);

    if (op == '+') {
        for (;i < vec_size ; i+=NB_DB)  {

            if (i + PREFETCH_DIST < vec_size) {
                // Pre-charged PREFETCH_DIST*8 bytes ahead
                _mm_prefetch((const char*)&v1[i+PREFETCH_DIST], _MM_HINT_T0);
                _mm_prefetch((const char*)&v2[i+PREFETCH_DIST], _MM_HINT_T0);
            }

            __m256d vec1 = _mm256_loadu_pd(&v1[i]);
            __m256d vec2 = _mm256_loadu_pd(&v2[i]);

            __m256d res = _mm256_add_pd(vec1, vec2);

            _mm256_storeu_pd(&new_data[i], res);
        }

        // Scalar residual
        for (; i < m*n; i++) {
            new_data[i] = v1[i] + v2[i];
        }
    }
    else if (op == '-') {
        for (;i < vec_size ; i+=NB_DB)  {

            if (i + PREFETCH_DIST < vec_size) {
                // Pre-charged PREFETCH_DIST*8 bytes ahead
                _mm_prefetch((const char*)&v1[i+PREFETCH_DIST], _MM_HINT_T0);
                _mm_prefetch((const char*)&v2[i+PREFETCH_DIST], _MM_HINT_T0);
            }

            __m256d vec1 = _mm256_loadu_pd(&v1[i]);
            __m256d vec2 = _mm256_loadu_pd(&v2[i]);

            __m256d res = _mm256_sub_pd(vec1, vec2);

            _mm256_storeu_pd(&new_data[i], res);
        }

        // Scalar residual
        for (; i < m*n; i++) {
            new_data[i] = v1[i] - v2[i];
        }
    }

    return new_data;
}

std::vector<double> multiply(const std::vector<double>& v1, const std::vector<double>& v2,
    size_t m, size_t n, size_t o, size_t p ) {
    
    // New data
    std::vector<double> new_data(m * p);
    
    // row - col
    size_t i = 0;
    size_t vec_sizei = m - (m % NB_DB);
    for (; i < vec_sizei; i+=NB_DB) {
        for (size_t j = 0; j < p; j++) {

            // Initialize variables
            size_t k = 0;
            size_t vec_size = n - (n % NB_DB);
            __m256d sum_vec0 = _mm256_setzero_pd();
            __m256d sum_vec1 = _mm256_setzero_pd();
            __m256d sum_vec2 = _mm256_setzero_pd();
            __m256d sum_vec3 = _mm256_setzero_pd();

            for (;k < vec_size; k+=NB_DB) {
                
                if (k + PREFETCH_DIST < vec_size) {
                    // Pre-charged PREFETCH_DIST*8 bytes ahead
                    if (i + PREFETCH_DIST1 < vec_sizei) {
                        for (size_t l = 0; l < NB_DB; l++) {
                            _mm_prefetch((const char*)&v1[(i+l+PREFETCH_DIST1) * n + k + PREFETCH_DIST], _MM_HINT_T0);
                        }
                    }
                    else {
                        _mm_prefetch((const char*)&v1[i * n + k + PREFETCH_DIST], _MM_HINT_T0);
                    }
                    _mm_prefetch((const char*)&v2[j * o + k + PREFETCH_DIST], _MM_HINT_T0);
                }

                // v1 row major
                // v2 col major
                __m256d vec1_0 = _mm256_loadu_pd(&v1[i * n + k]);
                __m256d vec1_1 = _mm256_loadu_pd(&v1[(i+1) * n + k]);
                __m256d vec1_2 = _mm256_loadu_pd(&v1[(i+2) * n + k]);
                __m256d vec1_3 = _mm256_loadu_pd(&v1[(i+3) * n + k]);

                __m256d vec2 = _mm256_loadu_pd(&v2[j * o + k]); 
                
                sum_vec0 = _mm256_fmadd_pd(vec1_0, vec2, sum_vec0);
                sum_vec1 = _mm256_fmadd_pd(vec1_1, vec2, sum_vec1);
                sum_vec2 = _mm256_fmadd_pd(vec1_2, vec2, sum_vec2);
                sum_vec3 = _mm256_fmadd_pd(vec1_3, vec2, sum_vec3);
            }

            // Horizontal Reduction
            double sum0 = horizontal_red(sum_vec0);
            double sum1 = horizontal_red(sum_vec1);
            double sum2 = horizontal_red(sum_vec2);
            double sum3 = horizontal_red(sum_vec3);
            
            // Scalar residual
            for (; k < n; k++) {
                sum0 += v1[i * n + k] * v2[j * o + k];
                sum1 += v1[(i+1) * n + k] * v2[j * o + k];
                sum2 += v1[(i+2) * n + k] * v2[j * o + k];
                sum3 += v1[(i+3) * n + k] * v2[j * o + k];
            }
            // Write it directly in col major
            new_data[j * m + i] = sum0;
            new_data[j * m + i+1] = sum1;
            new_data[j * m + i+2] = sum2;
            new_data[j * m + i+3] = sum3;
        }
    }

    // Scalar residual for i
    for (; i < m; i++) {
        for (size_t j = 0; j < p; j++) {

            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                // v1 row major
                // v2 col major
                sum += v1[i * n + k] * v2[j * o + k];
            }
            // Write it directly in col major
            new_data[j * m + i] = sum;
        }
    }
    // col - row
    // Non-optimized for AVX2 
    // ie need to access new_data[j*m + i], new_data[(j+1)*m + i], ...
    // Loss of our advantage
    
    // Return column - major
    return new_data;
}

std::vector<double> transpose(const std::vector<double>& v1,  
    size_t v1_rows, size_t v1_cols) {

    std::vector<double> new_data = Dataframe::transpose_blocks_avx2(v1_rows, v1_cols, v1);

    return new_data;
}

std::vector<double> inverse(const std::vector<double>& v1, size_t n,
    std::vector<double> swaps, std::vector<double> LU) {

    // Id matrix
    std::vector<double> id(n*n, 0.0);
    for (size_t i = 0; i < n; i++) {
        id[i*n + i] = 1;
    }

    // If no LU matrix was returned, then the matrix is triangular
    if (LU.empty()) {
        std::vector<double> y(n*n, 0.0);

        // Diag
        if (swaps[0] == 3) {
            for (size_t i = 0; i < n; i++) {
               y[i*n + i] = 1 / v1[i*n+i];
            }
            return y;
        }
        // Up
        else if (swaps[0] == 2) {

            // Solving Uy = Id 
            // Backward substitution by blocks
            size_t k = 0;
            size_t vec_size = n - (n % NB_DB);
            for (; k < vec_size; k+=NB_DB) {
                for (int i = static_cast<int>(n)-1; i >= 0; i--) {

                    double diag_i = v1[i*n + i];

                    __m256d sum = _mm256_set_pd(
                        id[(k+3)*n + i],
                        id[(k+2)*n + i],
                        id[(k+1)*n + i],
                        id[(k+0)*n + i]
                    );
                    
                    for (size_t j = i+1; j < n; j++) {
                        
                        __m256d LU_vec = _mm256_set1_pd(v1[j*n + i]);

                        __m256d y_vals = _mm256_set_pd(
                            y[(k+3)*n + j],
                            y[(k+2)*n + j],
                            y[(k+1)*n + j],
                            y[(k+0)*n + j]
                        );
                        
                        sum = _mm256_fnmadd_pd(LU_vec, y_vals, sum);
                    }
                    
                    double result[4];
                    _mm256_storeu_pd(result, sum);
                    
                    y[(k+0)*n + i] = result[0];
                    y[(k+1)*n + i] = result[1];
                    y[(k+2)*n + i] = result[2];
                    y[(k+3)*n + i] = result[3];

                    for(size_t p = 0; p < NB_DB; p++) {
                        if (std::abs(y[(k+p)*n + i]) < 1e-14) y[(k+p)*n + i] = 0;
                        else y[(k+p)*n + i] /= diag_i;
                    }
                }
            }

            // Scalar Residual for k 
            for (; k < n; k++) {
                for (int i = static_cast<int>(n)-1; i >= 0; i--) {

                    y[k*n + i] = id[k*n + i];
                    for (size_t j = i+1; j < n; j++) {
                        y[k*n + i] -= v1[j*n + i] * y[k*n + j];
                    }
                    if (std::abs(y[k*n + i]) < 1e-14) y[k*n + i] = 0;
                    else y[k*n + i] /= v1[i*n+i];
                }
            }
            return y;
        }
        // Down
        else {

            // Solving Ly = Id
            // Forward substitution by blocks with AVX2
            size_t k = 0;
            size_t vec_size = n - (n % NB_DB);

            for (; k < vec_size; k+=NB_DB) {

                for (size_t i = 0; i < n; i++) {

                    double diag_i = v1[i*n + i];

                    __m256d sum = _mm256_set_pd(
                        v1[(k+3)*n + i],
                        v1[(k+2)*n + i],
                        v1[(k+1)*n + i],
                        v1[(k+0)*n + i]
                    );
                    
                    for (size_t j = 0; j < i; j++) {

                        __m256d LU_val = _mm256_set1_pd(v1[j*n + i]);
                        
                        __m256d y_vals = _mm256_set_pd(
                            y[(k+3)*n + j],
                            y[(k+2)*n + j],
                            y[(k+1)*n + j],
                            y[(k+0)*n + j]
                        );
                        sum = _mm256_fnmadd_pd(LU_val, y_vals, sum);
                    }
                    
                    double result[4];
                    _mm256_storeu_pd(result, sum);

                    y[(k+0)*n + i] = result[0];
                    y[(k+1)*n + i] = result[1];
                    y[(k+2)*n + i] = result[2];
                    y[(k+3)*n + i] = result[3];

                    for(size_t p = 0; p < NB_DB; p++) {
                        if (std::abs(y[(k+p)*n + i]) < 1e-14) y[(k+p)*n + i] = 0;
                        else y[(k+p)*n + i] /= diag_i;
                    }
                }
            }

            // Scalar Residual for k
            for (; k < n; k++) {
                for (size_t i = 0; i < n; i++) {

                    double sum = 0.0;
                    sum = id[k*n + i];
                    for (size_t j = 0; j < i; j++) {
                        sum -= v1[j*n + i] * y[k*n + j];
                    }
                    y[k*n + i] = sum / v1[i*n + i];
                }
            }
            return y;
        }
    }
    else {
        // Permutation matrix
        swaps = transpose(swaps, n, n);

        // Row - Col to use multiply from AVX2
        std::vector<double> perm = multiply(swaps, id, n, n, n, n); 

        std::vector<double> res = solveLU_inplace(perm, LU, n);
        return res;
    }
}

#endif
}