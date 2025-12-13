#include "Linalg/LinalgAVX2.hpp"
#include "Linalg/Linalg.hpp"

namespace Linalg {
namespace AVX2 {
#ifdef __AVX2__

std::tuple<int, std::vector<double>, Dataframe> LU_decomposition(Dataframe& df) {

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
        auto [max, idx] = std::tuple{-1, 0};
        for (size_t i = k; i < n; i++) {

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
        size_t i = k+1, j = k+1;
        size_t vec_size = n - (n % NB_DB);

        alignas(32) double mult[4];
        
        for (; i < vec_size; i+=NB_DB) {

            // Prefectch if possible
            if (i + PREFETCH_DIST < vec_size) {
                _mm_prefetch((const char*)&LU[k*n + i + PREFETCH_DIST], _MM_HINT_T0);
            }

            __m256d vec_LU = _mm256_loadu_pd(&LU[k*n + i]);
            
            // Broadcast our pivot in vector
            __m256d pivot = _mm256_set1_pd(p);

            // L value 
            __m256d vec_Lvalue = _mm256_div_pd(vec_LU, pivot); 
            _mm256_storeu_pd(&LU[k*n + i], vec_Lvalue);
            
            // Update value in other cols 
            // Block version
            // Same method as Transpose_blocks_avx2
            for (; j < vec_size; j+=NB_DB) {
                
                if (j + df.PREFETCH_DIST1 < vec_size) {
                    for (size_t l = 0; l < NB_DB; l++) {
                        _mm_prefetch((const char*)&LU[(j + l + df.PREFETCH_DIST1)*n + i], _MM_HINT_T0);
                    }
                }

                __m256d val0 = _mm256_set1_pd(LU[j*n + k]);
                __m256d val1 = _mm256_set1_pd(LU[(j+1)*n + k]);
                __m256d val2 = _mm256_set1_pd(LU[(j+2)*n + k]);
                __m256d val3 = _mm256_set1_pd(LU[(j+3)*n + k]);

                __m256d vec_LU0 = _mm256_loadu_pd(&LU[j*n + i]);
                __m256d vec_LU1 = _mm256_loadu_pd(&LU[(j+1)*n + i]);
                __m256d vec_LU2 = _mm256_loadu_pd(&LU[(j+2)*n + i]);
                __m256d vec_LU3 = _mm256_loadu_pd(&LU[(j+3)*n + i]);

                // (a*b) - c
                vec_LU0 = _mm256_fnmadd_pd(val0, vec_Lvalue, vec_LU0); 
                vec_LU1 = _mm256_fnmadd_pd(val1, vec_Lvalue, vec_LU1); 
                vec_LU2 = _mm256_fnmadd_pd(val2, vec_Lvalue, vec_LU2); 
                vec_LU3 = _mm256_fnmadd_pd(val3, vec_Lvalue, vec_LU3); 

                _mm256_storeu_pd(&LU[j*n + i], vec_LU0);
                _mm256_storeu_pd(&LU[(j+1)*n + i], vec_LU1);
                _mm256_storeu_pd(&LU[(j+2)*n + i], vec_LU2);
                _mm256_storeu_pd(&LU[(j+3)*n + i], vec_LU3);
            }


            // Scalar residual
            if (j < n) {
                _mm256_storeu_pd(mult, vec_Lvalue);
            }
            for (; j < n; j++) {
                double pivot_val = LU[j*n + k];
                LU[j*n + i] -= mult[0] * pivot_val;
                LU[j*n + i+1] -= mult[1] * pivot_val;
                LU[j*n + i+2] -= mult[2] * pivot_val;
                LU[j*n + i+3] -= mult[3] * pivot_val;
            }
            j = k+1;
        }

        // Scalar residual
        for (; i < n; i++) {

            // L value 
            double mult = LU[k*n + i] / p;
            LU[k*n + i] = mult;

            // Update value in other cols 
            for (size_t j = k+1; j < n; j++) {
                LU[j*n + i] -= mult * LU[j*n + k];
            }
        }
    }
    
    return {nb_swaps, swaps, {n, n, false,  std::move(LU)}};
}

Dataframe solveLU_inplace(const Dataframe& perm, Dataframe& LU) {

    size_t n = LU.get_cols();
    std::vector<double> y(n*n);

    // Store the Diag
    std::vector<double> diag_U(n);
    for (size_t i = 0; i < n; i++) {
        diag_U[i] = LU.at(i * n + i);
    }

    size_t k = 0, i = 0;
    size_t vec_size = n - (n % NB_DB);

    // By Blocks with AVX2
    for (; k < vec_size; k+=NB_DB) {

        // Solving Ly = perm (No division because diag = 1)
        // Forward substitution 
        for (; i < vec_size; i+=NB_DB) {
            
            if (k + LU.PREFETCH_DIST1 < vec_size) {
                for (size_t p = 0; p < NB_DB; p++) {
                    _mm_prefetch((const char*)&perm.at((k+p+LU.PREFETCH_DIST1)*n + i), _MM_HINT_T0);
                }        
            }

            __m256d sum0 = _mm256_loadu_pd(&perm.at((k+0)*n + i));
            __m256d sum1 = _mm256_loadu_pd(&perm.at((k+1)*n + i));
            __m256d sum2 = _mm256_loadu_pd(&perm.at((k+2)*n + i));
            __m256d sum3 = _mm256_loadu_pd(&perm.at((k+3)*n + i));

            for (size_t j = 0; j < i; j++) {

                if (j + LU.PREFETCH_DIST1 < i) {
                    _mm_prefetch((const char*)&LU.at((j+LU.PREFETCH_DIST1)*n + i), _MM_HINT_T0);
                }

                __m256d LU_vec = _mm256_loadu_pd(&LU.at(j*n + i));

                __m256d y0 = _mm256_set1_pd(y[(k+0)*n + j]);
                __m256d y1 = _mm256_set1_pd(y[(k+1)*n + j]);
                __m256d y2 = _mm256_set1_pd(y[(k+2)*n + j]);
                __m256d y3 = _mm256_set1_pd(y[(k+3)*n + j]);

                sum0 = _mm256_fnmadd_pd(LU_vec, y0, sum0);
                sum1 = _mm256_fnmadd_pd(LU_vec, y1, sum1);
                sum2 = _mm256_fnmadd_pd(LU_vec, y2, sum2);
                sum3 = _mm256_fnmadd_pd(LU_vec, y3, sum3);
            }

            _mm256_storeu_pd(&y[k*n + i], sum0);
            _mm256_storeu_pd(&y[(k+1)*n + i], sum1);
            _mm256_storeu_pd(&y[(k+2)*n + i], sum2);
            _mm256_storeu_pd(&y[(k+3)*n + i], sum3);
        }

        // Scalar residual for i 
        for (; i < n; i++) {

            double sum0 = perm.at(k*n + i);
            double sum1 = perm.at((k+1)*n + i);
            double sum2 = perm.at((k+2)*n + i);
            double sum3 = perm.at((k+3)*n + i);

            for (size_t j = 0; j < i; j++) {
                sum0 -= LU.at(j*n + i) * y[k*n + j];
                sum1 -= LU.at(j*n + i) * y[(k+1)*n + j];
                sum2 -= LU.at(j*n + i) * y[(k+2)*n + j];
                sum3 -= LU.at(j*n + i) * y[(k+3)*n + j];
            }
            y[k*n + i] = sum0;
            y[(k+1)*n + i] = sum1;
            y[(k+2)*n + i] = sum2;
            y[(k+3)*n + i] = sum3;
        }

        // Solving Ux = y 
        // Backward substitution (Not worth in AVX2 due to i decreasing loop)
        for (int i = static_cast<int>(n)-1; i >= 0; i--) {

            double sum0 = y[k*n + i];
            double sum1 = y[(k+1)*n + i];
            double sum2 = y[(k+2)*n + i];
            double sum3 = y[(k+3)*n + i];

            for (size_t j = i+1; j < n; j++) {
                sum0 -= LU.at(j*n + i) * y[k*n + j];
                sum1 -= LU.at(j*n + i) * y[(k+1)*n + j];
                sum2 -= LU.at(j*n + i) * y[(k+2)*n + j];
                sum3 -= LU.at(j*n + i) * y[(k+3)*n + j];
            }

            y[k*n + i] = sum0;
            y[(k+1)*n + i] = sum1;
            y[(k+2)*n + i] = sum2;
            y[(k+3)*n + i] = sum3;

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
            sum = perm.at(k*n + i);
            for (size_t j = 0; j < i; j++) {
                sum -= LU.at(j*n + i) * y[k*n + j];
            }
            y[k*n + i] = sum;
        }

        // Solving Ux = y 
        // Backward substitution
        for (int i = static_cast<int>(n)-1; i >= 0; i--) {

            double sum = y[k*n + i];
            for (size_t j = i+1; j < n; j++) {
                sum -= LU.at(j*n + i) * y[k*n + j];
            }
            y[k*n + i] = sum;
            if (std::abs(y[k*n + i]) < 1e-14) y[k*n + i] = 0;
            else y[k*n + i] /= diag_U[i];
            
        }
    }
    return {n, n, false, std::move(y)};
}

double horizontal_red(__m256d& vec) {
    // hadd1 = [a+b, a+b, c+d, c+d] 
    __m256d hadd1 = _mm256_hadd_pd(vec, vec); 

    // sum128 = [a+b+c+d, ...]
    __m128d sum128 = _mm_add_pd(_mm256_castpd256_pd128(hadd1),  // [a+b, a+b]
                                _mm256_extractf128_pd(hadd1, 1));  // [c+d, c+d]
    
    // Extract result
    return _mm_cvtsd_f64(sum128);
}

Dataframe sum(Dataframe& df1, Dataframe& df2, char op) {

    size_t m = df1.get_rows();
    size_t n = df1.get_cols();
    size_t o = df2.get_rows();
    size_t p = df2.get_cols();

    // Verify if we can sum them
    if (m != o || n != p) throw std::runtime_error("Need two Matrix of equal dimensions");

    // Condition to have better performances
    if (df1.get_storage() != df2.get_storage()) {
        throw std::runtime_error("Need two Matrix with the same storage Col-major or Row-major for performances purpose");
    }

    // New data
    std::vector<double> new_data(m * n);
    
    size_t i = 0;
    size_t vec_size = m*n - ((m*n) % NB_DB);

    if (op == '+') {
        for (;i < vec_size ; i+=NB_DB)  {

            if (i + PREFETCH_DIST < vec_size) {
                // Pre-charged PREFETCH_DIST*8 bytes ahead
                _mm_prefetch((const char*)&df1.at(i+PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)&df2.at(i+PREFETCH_DIST), _MM_HINT_T0);
            }

            __m256d vec1 = _mm256_loadu_pd(&df1.at(i));
            __m256d vec2 = _mm256_loadu_pd(&df2.at(i));

            __m256d res = _mm256_add_pd(vec1, vec2);

            _mm256_storeu_pd(&new_data[i], res);
        }

        // Scalar residual
        for (; i < m*n; i++) {
            new_data[i] = df1.at(i) + df2.at(i);
        }
    }
    else if (op == '-') {
        for (;i < vec_size ; i+=NB_DB)  {

            if (i + PREFETCH_DIST < vec_size) {
                // Pre-charged PREFETCH_DIST*8 bytes ahead
                _mm_prefetch((const char*)&df1.at(i+PREFETCH_DIST), _MM_HINT_T0);
                _mm_prefetch((const char*)&df2.at(i+PREFETCH_DIST), _MM_HINT_T0);
            }

            __m256d vec1 = _mm256_loadu_pd(&df1.at(i));
            __m256d vec2 = _mm256_loadu_pd(&df2.at(i));

            __m256d res = _mm256_sub_pd(vec1, vec2);

            _mm256_storeu_pd(&new_data[i], res);
        }

        // Scalar residual
        for (; i < m*n; i++) {
            new_data[i] = df1.at(i) - df2.at(i);
        }
    }

    return {m, n, false, std::move(new_data)};
}

Dataframe multiply(Dataframe& df1, Dataframe& df2) {

    size_t m = df1.get_rows();
    size_t n = df1.get_cols();
    size_t o = df2.get_rows();
    size_t p = df2.get_cols();
    
    // Verify if we can multiply them
    if (n != o) throw std::runtime_error("Need df1 cols == df2 rows");

    // To optimize we want only row - col config (see explication at end of function)
    if (!(df1.get_storage() && !df2.get_storage())) throw std::runtime_error("Need df1 row major and df2 col major");

    std::vector<double> data(m * p, 0.0);
    
    // row - col
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {

            // Initialize variables
            size_t k = 0;
            size_t vec_size = n - (n % NB_DB);
            __m256d sum_vec = _mm256_setzero_pd();

            for (;k < vec_size; k+=NB_DB) {
                
                if (k + PREFETCH_DIST < vec_size) {
                    // Pre-charged PREFETCH_DIST*8 bytes ahead
                    _mm_prefetch((const char*)&df1.at(k+PREFETCH_DIST), _MM_HINT_T0);
                    _mm_prefetch((const char*)&df2.at(k+PREFETCH_DIST), _MM_HINT_T0);
                }

                // df1 row major
                // df2 col major
                __m256d vec1 = _mm256_loadu_pd(&df1.at(i * n + k));
                __m256d vec2 = _mm256_loadu_pd(&df2.at(j * o + k)); 
                
                __m256d res = _mm256_mul_pd(vec1, vec2);

                sum_vec = _mm256_add_pd(sum_vec, res);
            }

            // Horizontal Reduction
            double sum = horizontal_red(sum_vec);
            
            // Scalar residual
            for (; k < n; k++) {
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
        df.change_layout_inplace("AVX2");
    }

    std::vector<double> data = Dataframe::transpose_blocks_avx2(temp_row, temp_col, df.get_data(), NB_DB);

    return {rows, cols, false, std::move(data), df.get_headers(), 
        df.get_encoder(), df.get_encodedCols()};
}

Dataframe inverse(Dataframe& df) {

    auto [det, swaps, LU] = determinant(df);

    size_t n = df.get_cols();
    
    // Id matrix
    std::vector<double> id(n*n, 0.0);
    for (size_t i = 0; i < n; i++) {
        id[i*n + i] = 1;
    }
    Dataframe df_id = {n, n, false, std::move(id)};

    // If no LU matrix was returned, then the matrix is triangular
    if (LU.get_data().empty()) {
        std::vector<double> y(n*n, 0.0);

        // Diag
        if (swaps[0] == 3) {
            std::vector<double> y(n*n, 0.0);
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

                    double sum0 = df_id.at(k*n + i);
                    double sum1 = df_id.at((k+1)*n + i);
                    double sum2 = df_id.at((k+2)*n + i);
                    double sum3 = df_id.at((k+3)*n + i);

                    for (size_t j = i+1; j < n; j++) {
                        sum0 -= df.at(j*n + i) * y[k*n + j];
                        sum1 -= df.at(j*n + i) * y[(k+1)*n + j];
                        sum2 -= df.at(j*n + i) * y[(k+2)*n + j];
                        sum3 -= df.at(j*n + i) * y[(k+3)*n + j];
                    }

                    y[k*n + i] = sum0;
                    y[(k+1)*n + i] = sum1;
                    y[(k+2)*n + i] = sum2;
                    y[(k+3)*n + i] = sum3;

                    for(size_t p = 0; p < NB_DB; p++) {
                        if (std::abs(y[(k+p)*n + i]) < 1e-14) y[(k+p)*n + i] = 0;
                        else y[(k+p)*n + i] /= df.at(i*n+i);
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
            size_t k = 0, i = 0;
            size_t vec_size = n - (n % NB_DB);

            for (; i < vec_size; i+=NB_DB) {
            
                if (k + df.PREFETCH_DIST1 < vec_size) {
                    for (size_t p = 0; p < NB_DB; p++) {
                        _mm_prefetch((const char*)&df_id.at((k+p+df.PREFETCH_DIST1)*n + i), _MM_HINT_T0);
                    }        
                }

                __m256d sum0 = _mm256_loadu_pd(&df_id.at((k+0)*n + i));
                __m256d sum1 = _mm256_loadu_pd(&df_id.at((k+1)*n + i));
                __m256d sum2 = _mm256_loadu_pd(&df_id.at((k+2)*n + i));
                __m256d sum3 = _mm256_loadu_pd(&df_id.at((k+3)*n + i));

                for (size_t j = 0; j < i; j++) {

                    if (j + LU.PREFETCH_DIST1 < i) {
                        _mm_prefetch((const char*)&df.at((j+df.PREFETCH_DIST1)*n + i), _MM_HINT_T0);
                    }

                    __m256d df_vec = _mm256_loadu_pd(&df.at(j*n + i));

                    __m256d y0 = _mm256_set1_pd(y[(k+0)*n + j]);
                    __m256d y1 = _mm256_set1_pd(y[(k+1)*n + j]);
                    __m256d y2 = _mm256_set1_pd(y[(k+2)*n + j]);
                    __m256d y3 = _mm256_set1_pd(y[(k+3)*n + j]);

                    sum0 = _mm256_fnmadd_pd(df_vec, y0, sum0);
                    sum1 = _mm256_fnmadd_pd(df_vec, y1, sum1);
                    sum2 = _mm256_fnmadd_pd(df_vec, y2, sum2);
                    sum3 = _mm256_fnmadd_pd(df_vec, y3, sum3);
                }

                __m256d df_diag = _mm256_set_pd(df.at((i+3)*n + i+3), df.at((i+2)*n + i+2), 
                                                df.at((i+1)*n + i+1), df.at(i*n + i));

                _mm256_storeu_pd(&y[k*n + i], _mm256_div_pd(sum0, df_diag));
                _mm256_storeu_pd(&y[(k+1)*n + i], _mm256_div_pd(sum1, df_diag));
                _mm256_storeu_pd(&y[(k+2)*n + i], _mm256_div_pd(sum2, df_diag));
                _mm256_storeu_pd(&y[(k+3)*n + i], _mm256_div_pd(sum3, df_diag));
            }

            // Scalar residual for i 
            for (; i < n; i++) {

                double sum0 = df_id.at(k*n + i);
                double sum1 = df_id.at((k+1)*n + i);
                double sum2 = df_id.at((k+2)*n + i);
                double sum3 = df_id.at((k+3)*n + i);

                for (size_t j = 0; j < i; j++) {
                    sum0 -= df.at(j*n + i) * y[k*n + j];
                    sum1 -= df.at(j*n + i) * y[(k+1)*n + j];
                    sum2 -= df.at(j*n + i) * y[(k+2)*n + j];
                    sum3 -= df.at(j*n + i) * y[(k+3)*n + j];
                }
                y[k*n + i] = sum0 / df.at(i*n + i);
                y[(k+1)*n + i] = sum1 / df.at(i*n + i);
                y[(k+2)*n + i] = sum2 / df.at(i*n + i);
                y[(k+3)*n + i] = sum3 / df.at(i*n + i);
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
        df_swaps.change_layout_inplace();

        // Row - Col to use multiply from AVX2
        Dataframe perm = multiply(df_swaps, df_id); 

        return solveLU_inplace(perm, LU);
    }
}

#endif
}
}