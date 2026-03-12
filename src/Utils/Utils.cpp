#include <algorithm>
#include "Utils/Utils.hpp"

namespace Utils {
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

std::vector<double> compute_ranges(const std::vector<double>& data, size_t n_rows, size_t n_cols, 
    const std::vector<bool>& is_categorical, bool is_row_major) {
    
    std::vector<double> ranges(n_cols, 0.0);
    for (size_t j = 0; j < n_cols; j++) {
        if (is_categorical[j]) continue;

        double min_val = std::numeric_limits<double>::max();
        double max_val = std::numeric_limits<double>::lowest();

        for (size_t i = 0; i < n_rows; i++) {
            double val = is_row_major ? data[i * n_cols + j] : data[j * n_rows + i];
            if (!std::isnan(val)) {
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
        }
        // range = 0 if all values are identical
        ranges[j] = (max_val > min_val) ? (max_val - min_val) : 0.0;
    }
    return ranges;
}

bool allIntegers(const std::vector<const double*>& col) {
    return std::all_of(col.begin(), col.end(), [](const double* v) {
        return std::isnan(*v) || (*v == std::floor(*v));
    });
}
}