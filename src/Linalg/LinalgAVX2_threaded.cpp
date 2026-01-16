#include "Linalg/LinalgAVX2_threaded.hpp"
#include "Linalg/LinalgAVX2.hpp"
#include "Linalg/Linalg.hpp"

namespace Linalg {
namespace AVX2_threaded {
#ifdef __AVX2__

std::tuple<int, std::vector<double>, std::vector<double>> LU_decomposition(const Dataframe& df) {

    int nb_swaps = 0;
    size_t n = df.get_cols();
    std::vector<double> LU = df.get_data();

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

        // After multiple tests a minimum was decided
        if (n > 512 && n - k - 1 > 8) {
            // ThreadPool Variables
            ThreadPool& pool = ThreadPool::instance();
            size_t nb_threads = pool.nb_threads;

            std::vector<std::future<void>> futures;
            futures.reserve(nb_threads);

            size_t chunk = (int)((n - k - 1) / nb_threads);
            size_t start = k+1, end = k + 1 + chunk;

            for (size_t nb = 0; nb < nb_threads; nb++) {
                if (nb+1 == nb_threads) end = n;

                auto fut = pool.enqueue([start, end, k, n, p, vec_size, &LU] {
                    for (size_t i = start; i < end; i++) {

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
                });
                futures.push_back(std::move(fut));
                start += chunk;
                end += chunk;
            }
            for (auto& fut : futures) fut.wait();
        } else {
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
    }
    return std::make_tuple(nb_swaps, std::move(swaps), std::move(LU));
}

Dataframe solveLU_inplace(const std::vector<double>& perm, const std::vector<double>& LU, size_t n) {

    // After multiple tests a minimum was decided
    constexpr size_t THREADING_THRESHOLD = 512;
    if (n < THREADING_THRESHOLD) {
        return AVX2::solveLU_inplace(perm, LU, n);
    }

    std::vector<double> y(n*n, 0.0);
    size_t prefetch_dist = NB_DB;

    // Store the Diag
    std::vector<double> diag_U(n);
    for (size_t i = 0; i < n; i++) {
        diag_U[i] = LU[i * n + i];
    }

    size_t vec_size = n - (n % NB_DB);

    // ThreadPool Variables
    ThreadPool& pool = ThreadPool::instance();
    size_t nb_threads = pool.nb_threads;

    std::vector<std::future<void>> futures;
    futures.reserve(nb_threads);

    size_t chunk = (int)(vec_size / nb_threads);
    size_t start = 0, end = chunk;

    for (size_t nb = 0; nb < nb_threads; nb++) {
        if (nb+1 == nb_threads) end = vec_size;

        auto fut = pool.enqueue([start, end, n, &perm, &y, &diag_U, &LU] {
            
            // By Blocks with AVX2
            for (size_t k = start; k < end; k+=NB_DB) {

                // Solving Ly = perm (No division because diag = 1)
                // Forward substitution 
                for (size_t i = 0; i < n; i++) {

                    if (k + prefetch_dist < end) {
                        for (size_t p = 0; p < NB_DB; p++) {
                            _mm_prefetch((const char*)&perm[(k+p+prefetch_dist)*n + i], _MM_HINT_T0);
                        }        
                    }

                    __m256d sum = _mm256_set_pd(
                        perm[(k+3)*n + i],
                        perm[(k+2)*n + i],
                        perm[(k+1)*n + i],
                        perm[(k+0)*n + i]
                    );
                    
                    for (size_t j = 0; j < i; j++) {

                        if (j + prefetch_dist < i) {
                            for (size_t p = 0; p < NB_DB; p++) {
                                _mm_prefetch((const char*)&y[(k+p)*n + j + prefetch_dist], _MM_HINT_T0);
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
        });
        futures.push_back(std::move(fut));
        start += chunk;
        end += chunk;
    }
    for (auto& fut : futures) fut.wait();

    // Scalar Residual for k 
    for (size_t k = end; k < n; k++) {

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
    return {n, n, false, std::move(y)};
}

double horizontal_red(__m256d& vec) {
    return AVX2::horizontal_red(vec);
}

Dataframe sum(const Dataframe& df1, const Dataframe& df2, char op) {

    size_t m = df1.get_rows();
    size_t n = df1.get_cols();
    size_t o = df2.get_rows();
    size_t p = df2.get_cols();

    // After multiple tests a minimum was decided
    constexpr size_t THREADING_THRESHOLD = 512;
    if (m < THREADING_THRESHOLD) {
        return AVX2::sum(df1, df2, op);
    }

    // Verify if we can sum them
    if (m != o || n != p) throw std::runtime_error("Need two Matrix of equal dimensions");

    // Condition to have better performances
    if (df1.get_storage() != df2.get_storage()) {
        throw std::runtime_error("Need two Matrix with the same storage Col-major or Row-major for performances purpose");
    }

    // New data
    std::vector<double> new_data(m * n);
    size_t vec_size = m*n - ((m*n) % NB_DB);

    // ThreadPool Variables
    ThreadPool& pool = ThreadPool::instance();
    size_t nb_threads = pool.nb_threads;

    std::vector<std::future<void>> futures;
    futures.reserve(nb_threads);

    size_t chunk = (int)(vec_size / nb_threads);
    size_t start = 0, end = chunk;

    if (op == '+') {
        for (size_t nb = 0; nb < nb_threads; nb++) {
            if (nb+1 == nb_threads) end = vec_size;

            auto fut = pool.enqueue([start, end, &df, &new_data] {
                for (size_t i = start; i < end; i += NB_DB) {

                    if (i + PREFETCH_DIST < end) {
                        // Pre-charged PREFETCH_DIST*8 bytes ahead
                        _mm_prefetch((const char*)&df1.at(i+PREFETCH_DIST), _MM_HINT_T0);
                        _mm_prefetch((const char*)&df2.at(i+PREFETCH_DIST), _MM_HINT_T0);
                    }

                    __m256d vec1 = _mm256_loadu_pd(&df1.at(i));
                    __m256d vec2 = _mm256_loadu_pd(&df2.at(i));

                    __m256d res = _mm256_add_pd(vec1, vec2);

                    _mm256_storeu_pd(&new_data[i], res);
                }
            });
            futures.push_back(std::move(fut));
            start += chunk;
            end += chunk;
        }

        for (auto& fut : futures) fut.wait();

        // Scalar residual
        for (size_t i = end; i < m*n; i++) {
            new_data[i] = df1.at(i) + df2.at(i);
        }
    }
    else if (op == '-') {
        for (size_t n = 0; n < nb_threads; n++) {
            if (n+1 == nb_threads) end = vec_size;

            auto fut = pool.enqueue([start, end, &df, &new_data] {
                for (size_t i = start; i < end; i += NB_DB) {

                    if (i + PREFETCH_DIST < end) {
                        // Pre-charged PREFETCH_DIST*8 bytes ahead
                        _mm_prefetch((const char*)&df1.at(i+PREFETCH_DIST), _MM_HINT_T0);
                        _mm_prefetch((const char*)&df2.at(i+PREFETCH_DIST), _MM_HINT_T0);
                    }

                    __m256d vec1 = _mm256_loadu_pd(&df1.at(i));
                    __m256d vec2 = _mm256_loadu_pd(&df2.at(i));

                    __m256d res = _mm256_sub_pd(vec1, vec2);

                    _mm256_storeu_pd(&new_data[i], res);
                }
            });
            futures.push_back(std::move(fut));
            start += chunk;
            end += chunk;
        }

        for (auto& fut : futures) fut.wait();

        // Scalar residual
        for (size_t i = end; i < m*n; i++) {
            new_data[i] = df1.at(i) + df2.at(i);
        }
    }

    return {m, n, false, std::move(new_data)};
}

Dataframe multiply(const Dataframe& df1, const Dataframe& df2) {

    size_t m = df1.get_rows();
    size_t n = df1.get_cols();
    size_t o = df2.get_rows();
    size_t p = df2.get_cols();

    // After multiple tests a minimum was decided
    constexpr size_t THREADING_THRESHOLD = 512;
    if (m < THREADING_THRESHOLD) {
        return AVX2::multiply(df1, df2);
    }
    
    // Verify if we can multiply them
    if (n != o) throw std::runtime_error("Need df1 cols == df2 rows");

    // To optimize we want only row - col config (see explication at end of function)
    if (!(df1.get_storage() && !df2.get_storage())) throw std::runtime_error("Need df1 row major and df2 col major");

    // row - col
    std::vector<double> data(m * p);
    size_t vec_sizei = m - (m % NB_DB);
    
    // ThreadPool Variables
    ThreadPool& pool = ThreadPool::instance();
    size_t nb_threads = pool.nb_threads;

    std::vector<std::future<void>> futures;
    futures.reserve(nb_threads);

    size_t chunk = (int)(vec_sizei / nb_threads);
    size_t start = 0, end = chunk;

    for (size_t nb = 0; nb < nb_threads; nb++) {
        if (nb+1 == nb_threads) end = vec_sizei;

        auto fut = pool.enqueue([start, end, m, n, o, p, &df, &data] {
            for (size_t i = start; i < end; i += NB_DB) {
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
                            if (i + df1.PREFETCH_DIST1 < end) {
                                for (size_t l = 0; l < NB_DB; l++) {
                                    _mm_prefetch((const char*)&df1.at((i+l+df1.PREFETCH_DIST1) * n + k + PREFETCH_DIST), _MM_HINT_T0);
                                }
                            }
                            else {
                                _mm_prefetch((const char*)&df1.at(i * n + k + PREFETCH_DIST), _MM_HINT_T0);
                            }
                            _mm_prefetch((const char*)&df2.at(j * o + k + PREFETCH_DIST), _MM_HINT_T0);
                        }

                        // df1 row major
                        // df2 col major
                        __m256d vec1_0 = _mm256_loadu_pd(&df1.at(i * n + k));
                        __m256d vec1_1 = _mm256_loadu_pd(&df1.at((i+1) * n + k));
                        __m256d vec1_2 = _mm256_loadu_pd(&df1.at((i+2) * n + k));
                        __m256d vec1_3 = _mm256_loadu_pd(&df1.at((i+3) * n + k));

                        __m256d vec2 = _mm256_loadu_pd(&df2.at(j * o + k)); 
                        
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
                        sum0 += df1.at(i * n + k) * df2.at(j * o + k);
                        sum1 += df1.at((i+1) * n + k) * df2.at(j * o + k);
                        sum2 += df1.at((i+2) * n + k) * df2.at(j * o + k);
                        sum3 += df1.at((i+3) * n + k) * df2.at(j * o + k);
                    }
                    // Write it directly in col major
                    data[j * m + i] = sum0;
                    data[j * m + i+1] = sum1;
                    data[j * m + i+2] = sum2;
                    data[j * m + i+3] = sum3;
                }
            }
        });
        futures.push_back(std::move(fut));
        start += chunk;
        end += chunk;
    }

    for (auto& fut : futures) fut.wait();

    // Scalar residual for i
    for (size_t i = end; i < m; i++) {
        for (size_t j = 0; j < p; j++) {

            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                // df1 row major
                // df2 col major
                sum += df1.at(i * n + k) * df2.at(j * o + k);
            }
            // Write it directly in col major
            data[j * m + i] = sum;
        }
    }
    // col - row
    // Non-optimized for AVX2 
    // ie need to access data[j*m + i], data[(j+1)*m + i], ...
    // Loss of our advantage
    
    // Return column - major
    return Dataframe(m, p, false, std::move(data), 
                     df1.get_headers(), df1.get_encoder(), df1.get_encodedCols());
}

Dataframe transpose(Dataframe& df) {

    size_t rows = df.get_cols(), cols = df.get_rows();
    size_t temp_row = df.get_rows(), temp_col = df.get_cols();

    // Changing layout for better performances later
    if (df.get_storage()){
        df.change_layout_inplace("AVX2_threaded");
    }

    std::vector<double> data = Dataframe::transpose_avx2_th(temp_row, temp_col, df.get_data());

    return {rows, cols, false, std::move(data), df.get_headers(), 
        df.get_encoder(), df.get_encodedCols()};
}

/*Dataframe inverse(Dataframe& df) {

    auto [det, swaps, LU] = determinant(df);

    size_t n = df.get_cols();
    
    // Id matrix
    std::vector<double> id(n*n, 0.0);
    for (size_t i = 0; i < n; i++) {
        id[i*n + i] = 1;
    }
    Dataframe df_id = {n, n, false, std::move(id)};

    // If no LU matrix was returned, then the matrix is triangular
    if (LU.empty()) {
        std::vector<double> y(n*n, 0.0);

        // Diag
        if (swaps[0] == 3) {
            for (size_t i = 0; i < n; i++) {
               y[i*n + i] = 1 / df.at(i*n+i);
            }
            return {n, n, false, std::move(y)};
        }
        // Up
        else if (swaps[0] == 2) {

            // Solving Uy = Id 
            // Backward substitution by blocks
            size_t k = 0;
            size_t vec_size = n - (n % NB_DB);
            for (; k < vec_size; k+=NB_DB) {
                for (int i = static_cast<int>(n)-1; i >= 0; i--) {

                    double diag_i = df.at(i*n + i);

                    __m256d sum = _mm256_set_pd(
                        df_id.at((k+3)*n + i),
                        df_id.at((k+2)*n + i),
                        df_id.at((k+1)*n + i),
                        df_id.at((k+0)*n + i)
                    );
                    
                    for (size_t j = i+1; j < n; j++) {
                        
                        __m256d LU_vec = _mm256_set1_pd(df.at(j*n + i));

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

                    y[k*n + i] = df_id.at(k*n + i);
                    for (size_t j = i+1; j < n; j++) {
                        y[k*n + i] -= df.at(j*n + i) * y[k*n + j];
                    }
                    if (std::abs(y[k*n + i]) < 1e-14) y[k*n + i] = 0;
                    else y[k*n + i] /= df.at(i*n+i);
                }
            }
            return {n, n, false, std::move(y)};
        }
        // Down
        else {

            // Solving Ly = Id
            // Forward substitution by blocks with AVX2
            size_t k = 0;
            size_t vec_size = n - (n % NB_DB);

            for (; k < vec_size; k+=NB_DB) {

                for (size_t i = 0; i < n; i++) {

                    double diag_i = df.at(i*n + i);

                    __m256d sum = _mm256_set_pd(
                        df_id.at((k+3)*n + i),
                        df_id.at((k+2)*n + i),
                        df_id.at((k+1)*n + i),
                        df_id.at((k+0)*n + i)
                    );
                    
                    for (size_t j = 0; j < i; j++) {

                        __m256d LU_val = _mm256_set1_pd(df.at(j*n + i));
                        
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
                    sum = df_id.at(k*n + i);
                    for (size_t j = 0; j < i; j++) {
                        sum -= df.at(j*n + i) * y[k*n + j];
                    }
                    y[k*n + i] = sum / df.at(i*n + i);
                }
            }
            return {n, n, false, std::move(y)};
        }
    }
    else {
        
        // Permutation matrix
        Dataframe df_swaps = {n, n, false, std::move(swaps)};
        df_swaps.change_layout_inplace("AVX2_threaded");

        // Row - Col to use multiply from AVX2
        Dataframe perm = multiply(df_swaps, df_id); 

        Dataframe res = solveLU_inplace(perm.get_data(), LU, n);
        return res;
    }
}*/

#endif
}
}