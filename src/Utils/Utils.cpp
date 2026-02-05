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
}