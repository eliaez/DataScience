#include "Linalg/LinalgAVX2.hpp"
#include "Linalg/Linalg.hpp"
#include <immintrin.h>

namespace Linalg {
namespace AVX2 {

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
    size_t vec_size = m*n - ((m*n) % 4);

    if (op == '+') {
        for (; i + PREFETCH_DIST < m*n && i < vec_size ; i+=NB_DB)  {

            // Pre-charged PREFETCH_DIST*8 bytes ahead
            _mm_prefetch((const char*)&df1.at(i+PREFETCH_DIST), _MM_HINT_T0);
            _mm_prefetch((const char*)&df2.at(i+PREFETCH_DIST), _MM_HINT_T0);

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
        for (; i + PREFETCH_DIST < m*n && i < vec_size ; i+=NB_DB)  {

            // Pre-charged PREFETCH_DIST*8 bytes ahead
            _mm_prefetch((const char*)&df1.at(i+PREFETCH_DIST), _MM_HINT_T0);
            _mm_prefetch((const char*)&df2.at(i+PREFETCH_DIST), _MM_HINT_T0);

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




}
}