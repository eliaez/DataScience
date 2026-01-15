#include "Linalg/LinalgAVX2_threaded.hpp"
#include "Linalg/LinalgAVX2.hpp"
#include "Linalg/Linalg.hpp"

namespace Linalg {
namespace AVX2_threaded {
#ifdef __AVX2__

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

#endif
}
}