#include "Linalg/LinalgAVX2.hpp"
#include "Linalg/Linalg.hpp"

namespace Linalg {
namespace AVX2 {

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

    std::vector<double> data(rows*cols);

    // Changing layout for better performances later
    if (df.get_storage()){
        df.change_layout_inplace();
    }

    // Variables
    size_t i = 0, j = 0;
    size_t vec_sizei = temp_row - (temp_row % NB_DB);
    size_t vec_sizej = temp_col - (temp_col % NB_DB);

    for (; i < vec_sizei; i += NB_DB) {
        for (; j < vec_sizej; j += NB_DB) {

            if (j + PREFETCH_DIST1 < vec_sizej) {
                _mm_prefetch(&df.at((j+PREFETCH_DIST1+0) * temp_row + i), _MM_HINT_T0);
                _mm_prefetch(&df.at((j+PREFETCH_DIST1+1) * temp_row + i), _MM_HINT_T0);
                _mm_prefetch(&df.at((j+PREFETCH_DIST1+2) * temp_row + i), _MM_HINT_T0);
                _mm_prefetch(&df.at((j+PREFETCH_DIST1+3) * temp_row + i), _MM_HINT_T0);
            }
            
            // Load 4 cols
            __m256d col0 = _mm256_loadu_pd(&df.at((j+0)*temp_row + i));
            __m256d col1 = _mm256_loadu_pd(&df.at((j+1)*temp_row + i));
            __m256d col2 = _mm256_loadu_pd(&df.at((j+2)*temp_row + i));
            __m256d col3 = _mm256_loadu_pd(&df.at((j+3)*temp_row + i));
            
            // Get pair elements of each
            __m256d t0 = _mm256_unpacklo_pd(col0, col1);

            // Get odd elements of each 
            __m256d t1 = _mm256_unpackhi_pd(col0, col1);

            __m256d t2 = _mm256_unpacklo_pd(col2, col3);
            __m256d t3 = _mm256_unpackhi_pd(col2, col3);
            
            // Get two first elements of each 
            __m256d row0 = _mm256_permute2f128_pd(t0, t2, 0x20);
            __m256d row1 = _mm256_permute2f128_pd(t1, t3, 0x20);

            // Get two last elements of each
            __m256d row2 = _mm256_permute2f128_pd(t0, t2, 0x31);
            __m256d row3 = _mm256_permute2f128_pd(t1, t3, 0x31);
            
            _mm256_storeu_pd(&data[(i+0)*temp_col + j], row0);
            _mm256_storeu_pd(&data[(i+1)*temp_col + j], row1);
            _mm256_storeu_pd(&data[(i+2)*temp_col + j], row2);
            _mm256_storeu_pd(&data[(i+3)*temp_col + j], row3);
        }

        // Scalar residual
        for(; j < temp_col; j++) {
                data[(i+0)*temp_col + j] = df.at(j*temp_row + (i+0));
                data[(i+1)*temp_col + j] = df.at(j*temp_row + (i+1));
                data[(i+2)*temp_col + j] = df.at(j*temp_row + (i+2));
                data[(i+3)*temp_col + j] = df.at(j*temp_row + (i+3));
        }
        j = 0;
    }

    // Scalar residual 
    for (; i < temp_row; i++) {
        for(size_t j = 0; j < temp_col; j++) {
            data[i*temp_col + j] = df.at(j*temp_row + i);
        }
    }   

    return {rows, cols, false, std::move(data), df.get_headers(), 
        df.get_encoder(), df.get_encodedCols()};
}

Dataframe inverse(Dataframe& df) {

    return {};
}

}
}