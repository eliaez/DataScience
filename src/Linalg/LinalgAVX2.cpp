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

    return {};
}

#endif
}
}