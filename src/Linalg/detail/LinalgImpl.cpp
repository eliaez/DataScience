#include "Linalg/detail/LinalgImpl.hpp"
#include <string>
#include <stdexcept>


// ============================================
// Macro Dispatch
// ============================================

#if defined(__AVX2__) && defined(USE_MKL)
    #define DISPATCH_BACKEND(func, ...) \
        switch(Operations::get_backend()) { \
            case Linalg::Backend::NAIVE: return Linalg::Naive::func(__VA_ARGS__); \
            case Linalg::Backend::AVX2: return Linalg::AVX2::func(__VA_ARGS__); \
            case Linalg::Backend::AVX2_THREADED: return Linalg::AVX2_threaded::func(__VA_ARGS__); \
            case Linalg::Backend::EIGEN: return Linalg::EigenNP::func(__VA_ARGS__); \
            case Linalg::Backend::MKL: return Linalg::MKL::func(__VA_ARGS__); \
            default: return Linalg::AVX2_threaded::func(__VA_ARGS__); \
        }

    #define DISPATCH_BACKEND2(func, ...) \
        switch(Operations::get_backend()) { \
            case Linalg::Backend::NAIVE: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::Naive::func(__VA_ARGS__); \
                break; \
            } \
            case Linalg::Backend::AVX2: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::AVX2::func(__VA_ARGS__); \
                break; \
            } \
            case Linalg::Backend::AVX2_THREADED: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::AVX2_threaded::func(__VA_ARGS__); \
                break; \
            } \
            default: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::AVX2_threaded::func(__VA_ARGS__); \
                break; \
            } \
        } 
#elif defined(__AVX2__)
    #define DISPATCH_BACKEND(func, ...) \
        switch(Operations::get_backend()) { \
            case Linalg::Backend::NAIVE: return Linalg::Naive::func(__VA_ARGS__); \
            case Linalg::Backend::AVX2: return Linalg::AVX2::func(__VA_ARGS__); \
            case Linalg::Backend::AVX2_THREADED: return Linalg::AVX2_threaded::func(__VA_ARGS__); \
            case Linalg::Backend::EIGEN: return Linalg::EigenNP::func(__VA_ARGS__); \
            default: return Linalg::AVX2_threaded::func(__VA_ARGS__); \
        }

    #define DISPATCH_BACKEND2(func, ...) \
        switch(Operations::get_backend()) { \
            case Linalg::Backend::NAIVE: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::Naive::func(__VA_ARGS__); \
                break; \
            } \
            case Linalg::Backend::AVX2: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::AVX2::func(__VA_ARGS__); \
                break; \
            } \
            case Linalg::Backend::AVX2_THREADED: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::AVX2_threaded::func(__VA_ARGS__); \
                break; \
            } \
            default: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::AVX2_threaded::func(__VA_ARGS__); \
                break; \
            } \
        } 
#elif defined(USE_MKL)
    #define DISPATCH_BACKEND(func, ...) \
        switch(Operations::get_backend()) { \
            case Linalg::Backend::NAIVE: return Linalg::Naive::func(__VA_ARGS__); \
            case Linalg::Backend::EIGEN: return Linalg::EigenNP::func(__VA_ARGS__); \
            case Linalg::Backend::MKL: return Linalg::MKL::func(__VA_ARGS__); \
            default: return Linalg::MKL::func(__VA_ARGS__); \
        }

    #define DISPATCH_BACKEND2(func, ...) \
        switch(Operations::get_backend()) { \
            case Linalg::Backend::NAIVE: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::Naive::func(__VA_ARGS__); \
                break; \
            } \
            default: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::Naive::func(__VA_ARGS__); \
                break; \
            } \
        } 
#else
    #define DISPATCH_BACKEND(func, ...) \
        switch(Operations::get_backend()) { \
            case Linalg::Backend::NAIVE: return Linalg::Naive::func(__VA_ARGS__); \
            case Linalg::Backend::EIGEN: return Linalg::EigenNP::func(__VA_ARGS__); \
            default: return Linalg::Naive::func(__VA_ARGS__); \
        }

    #define DISPATCH_BACKEND2(func, ...) \
        switch(Operations::get_backend()) { \
            case Linalg::Backend::NAIVE: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::Naive::func(__VA_ARGS__); \
                break; \
            } \
            default: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::Naive::func(__VA_ARGS__); \
                break; \
            } \
        } 
#endif

// ============================================
// Internal Implementation 
// ============================================

namespace Linalg {

std::vector<double> Operations::Impl::sum_impl(
    const std::vector<double>& v1, const std::vector<double>& v2,   // Data
    size_t v1_rows, size_t v1_cols, size_t v2_rows, size_t v2_cols, // Size
    bool v1_layout, bool v2_layout,                                 // Layouts
    char op ) {                                                     // Operator
    
    // Verify if we can sum them
    if (v1_rows != v2_rows || v1_cols != v2_cols) throw std::runtime_error("Need two Matrix of equal dimensions");
    
    // Condition to have better performance and match backend requirements
    const std::string backend_str = Linalg::get_backend();
    if (backend_str == "MKL" || backend_str == "Eigen") {

        if ((v1_layout != v2_layout) || v1_layout) {
            throw std::runtime_error("Need two Matrix with the same storage and Col-major for performance and backend purpose");
        }
    }
    else if (v1_layout != v2_layout) {
        throw std::runtime_error("Need two Matrix with the same storage Col-major or Row-major for performance and backend purpose");
    }

    // Dispatch
    DISPATCH_BACKEND(sum, v1, v2, v1_rows, v1_cols, op)
}

std::vector<double> Operations::Impl::multiply_impl(
    const std::vector<double>& v1, const std::vector<double>& v2,   // Data
    size_t v1_rows, size_t v1_cols, size_t v2_rows, size_t v2_cols, // Size
    bool v1_layout, bool v2_layout) {                               // Layouts
        
    // Verify if we can multiply them
    if (v1_cols != v2_rows) throw std::runtime_error("Need df1 cols == df2 rows");

    // Condition to have better performance and match backend requirements
    const std::string backend_str = Linalg::get_backend();
    if (backend_str == "MKL" || backend_str == "Eigen") {

        if ((v1_layout != v2_layout) || v1_layout) {
            throw std::runtime_error("Need two Matrix with the same storage and Col-major for performance and backend purpose");
        }
    }
    else if (!v1_layout || v2_layout) {
        throw std::runtime_error("Need df1 row major and df2 col major");
    }

    // Dispatch
    DISPATCH_BACKEND(multiply, v1, v2, v1_rows, v1_cols, v2_rows, v2_cols)
}

std::vector<double> Operations::Impl::transpose_impl(
    const std::vector<double>& v1,                          // Data
    size_t v1_rows, size_t v1_cols, bool v1_layout) {     // Size and layout

    if (v1_layout) {
        throw std::runtime_error("Need df col major");
    }
    
    // Dispatch
    DISPATCH_BACKEND(transpose, v1, v1_rows, v1_cols)
}

std::vector<double> Operations::Impl::inverse_impl(
    const std::vector<double>& v1,      // Data
    size_t v1_rows, bool v1_layout) {   // Size and layout 

    double det;
    std::vector<double> swaps;
    std::vector<double> LU;
    
    // MKL and Eigen are external libraries
    const std::string backend_str = Linalg::get_backend();
    if (backend_str != "MKL" && backend_str != "Eigen") {
        std::tie(det, swaps, LU) = determinant_impl(v1, v1_rows, v1_layout);
    } 
    else {
        det = 0.0;
        swaps = {};
        LU = {};
    }

    // Dispatch
    DISPATCH_BACKEND(inverse, v1, v1_rows, swaps, LU)
}

std::tuple<double, std::vector<double>, std::vector<double>> Operations::Impl::determinant_impl(
    const std::vector<double>& v1,      // Data
    size_t v1_rows, bool v1_layout) {   // Size and layout 

    if (v1_layout) {
        throw std::runtime_error("Need df col major");
    }
    
    // Variables
    const size_t n = v1_rows;  
    const std::string backend_str = Linalg::get_backend();
    
    // LU decomposition
    std::vector<double> LU;

    // Let's see if the matrix is diagonal or triangular 
    int test_v;
    test_v = triangular_impl(v1, n, n, v1_layout);
    if (test_v != 0) {
        
        double det = 1;
        for (size_t j = 0; j < n; j++) {
            
            det *= v1[j*n + j]; // Product of the diagonal
        }

        std::vector<double> vec_v = {static_cast<double>(test_v)};
        
        return std::make_tuple(det, std::move(vec_v), std::move(LU));
    }
    else {
        int nb_swaps;
        std::vector<double> swaps;

        DISPATCH_BACKEND2(LU_decomposition, v1, n)

        double det = (nb_swaps % 2) ? -1.0 : 1.0;
        for (size_t j = 0; j < n; j++) {
            
            det *= LU[j*n + j]; // Product of the diagonal
        }
        return std::make_tuple(det, std::move(swaps), std::move(LU));
    }
}

int Operations::Impl::triangular_impl(
    const std::vector<double>& v1,      // Data
    size_t v1_rows, size_t v1_cols, bool v1_layout) {   // Size and layout

    // Verify if we have a square matrix
    if (v1_rows != v1_cols) throw std::runtime_error("Need Matrix(n,n)");

    size_t n = v1_rows; 
    bool is_trig_up = true;
    bool is_trig_down = true;

    #ifdef __AVX2__
        // AVX2 variables
        size_t NB_DB = Linalg::AVX2::NB_DB;
        size_t PREFETCH_DIST = Linalg::AVX2::PREFETCH_DIST;
        __m256d zero_vec = _mm256_set1_pd(0.0);

        // Triangular inf in row major
        for (size_t i = 1; i < n && is_trig_down; i++) {
            
            size_t j = 0;
            size_t vec_end = (i / NB_DB) * NB_DB;  // Blocs complets
            
            for (; j < vec_end; j += NB_DB) {
                if (j + PREFETCH_DIST < vec_end) {
                    _mm_prefetch((const char*)&v1[i*n + j + PREFETCH_DIST], _MM_HINT_T0);
                }
                
                __m256d vec = _mm256_loadu_pd(&v1[i*n + j]);
                __m256d cmp = _mm256_cmp_pd(vec, zero_vec, _CMP_EQ_OQ);
                int mask = _mm256_movemask_pd(cmp);
                
                if (mask != 0xF) {
                    is_trig_down = false;
                    break;
                }
            }

            // Scalar residual for i
            for (; j < i && is_trig_down; j++) {
                if (v1[i*n + j] != 0.0) is_trig_down = false;
            }
        }

        // Triangular sup in row major 
        for (size_t i = 0; i < n-1 && is_trig_up; i++) {
            
            size_t j = i + 1;
            size_t vec_end = n - ((n - i - 1) % NB_DB);
            
            for (; j < vec_end; j += NB_DB) {
                if (j + PREFETCH_DIST < vec_end) {
                    _mm_prefetch((const char*)&v1[i*n + j + PREFETCH_DIST], _MM_HINT_T0);
                }
                
                __m256d vec = _mm256_loadu_pd(&v1[i*n + j]);
                __m256d cmp = _mm256_cmp_pd(vec, zero_vec, _CMP_EQ_OQ);
                int mask = _mm256_movemask_pd(cmp);
                
                if (mask != 0xF) {
                    is_trig_up = false;
                    break;
                }
            }

            // Scalar residual
            for (; j < n && is_trig_up; j++) {
                if (v1[i*n + j] != 0.0) is_trig_up = false;
            }
        }
    #else
        // Triangular inf in row major
        for (size_t j = 0; j < n && is_trig_down; j++) {
            for(size_t i = 0; i < j && is_trig_down; i++) {
                
                if (v1[j*n + i] != 0) is_trig_down = false;
            }
        }

        // Triangular sup in row major
        for (size_t j = 0; j < n && is_trig_up; j++) {
            for(size_t i = j+1; i < n && is_trig_up; i++) {
                
                if (v1[j*n + i] != 0) is_trig_up = false;
            }
        }
    #endif

    // We detected in row major config
    if (v1_layout) std::swap(is_trig_up, is_trig_down);

    if (is_trig_up && is_trig_down) return 3; // Diag
    else if (is_trig_up) return 2; // Up
    else if (is_trig_down) return 1; // Down

    return 0; // Not triangular
}
}