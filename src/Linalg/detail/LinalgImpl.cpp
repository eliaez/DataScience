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

namespace Linalg::detail {

std::vector<double> OperationsImpl::sum_impl(
    const std::vector<double>& v1, const std::vector<double>& v2,   // Data
    size_t v1_rows, size_t v1_cols, size_t v2_rows, size_t v2_cols, // Size
    bool v1_layout, bool v2_layout,                                 // Layouts
    char op ) {                                                     // Operator
    
    // Verify if we can sum them
    if (v1_rows != v2_rows || v1_cols != v2_cols) throw std::runtime_error("Need two Matrix of equal dimensions");
    
    // Condition to have better performance and match backend requirements
    const std::string backend = get_backend();
    if (backend == "MKL" || backend == "Eigen") {

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

std::vector<double> OperationsImpl::multiply_impl(
    const std::vector<double>& v1, const std::vector<double>& v2,   // Data
    size_t v1_rows, size_t v1_cols, size_t v2_rows, size_t v2_cols, // Size
    bool v1_layout, bool v2_layout) {                               // Layouts
        
    // Verify if we can multiply them
    if (v1_cols != v2_rows) throw std::runtime_error("Need df1 cols == df2 rows");

    // Condition to have better performance and match backend requirements
    const std::string backend = get_backend();
    if (backend == "MKL" || backend == "Eigen") {

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

std::vector<double> OperationsImpl::transpose_impl(
    const std::vector<double>& v1,      // Data
    size_t v1_rows, size_t v1_cols) {   // Size 
    
    // Dispatch
    DISPATCH_BACKEND(transpose, v1, v1_rows, v1_cols)
}

}