#include "Linalg/Linalg.hpp"

// ============================================
// Public Interface
// ============================================

namespace Linalg {

Backend Operations::current_backend = Backend::AUTO;

void Operations::set_backend(Backend b) {
    current_backend = b;
}

void Operations::set_backend(const std::string& b) {
    
    if (b == "Naive") current_backend = Backend::NAIVE;
    else if (b == "Eigen") current_backend = Backend::EIGEN;
    
    #if defined(__AVX2__) && defined(USE_MKL)
        else if (b == "AVX2") current_backend = Backend::AVX2;
        else if (b == "AVX2_threaded") current_backend = Backend::AVX2_THREADED;
        else if (b == "MKL") current_backend = Backend::MKL;
        else current_backend = Backend::AVX2_THREADED;
    #elif defined(__AVX2__)
        else if (b == "AVX2") current_backend = Backend::AVX2;
        else if (b == "AVX2_threaded") current_backend = Backend::AVX2_THREADED;
        else current_backend = Backend::AVX2_THREADED;
    #elif defined(USE_MKL)
        else if (b == "MKL") current_backend = Backend::MKL;
        else current_backend = Backend::NAIVE;
    #else 
        else current_backend = Backend::NAIVE;
    #endif 
}

Backend Operations::get_backend() {
    
    // Select the best one
    if (current_backend == Backend::AUTO) {

        // AVX2 for now
        #if defined(__AVX2__)
            return Backend::AVX2_THREADED;
        #else
            return Backend::NAIVE;
        #endif
    } 
    return current_backend;
}

Dataframe Operations::sum(const Dataframe& df1, const Dataframe& df2, char op) {
    
    size_t m = df1.get_rows();
    size_t n = df1.get_cols();
    bool is_row_major = df1.get_storage();

    std::vector<double> res = detail::OperationsImpl::sum_impl(
        df1.get_data(), df2.get_data(), 
        m, n,
        df2.get_rows(), df2.get_cols(),
        is_row_major, df2.get_storage(), 
        op
    );

    return Dataframe(m, n, is_row_major, std::move(res));
}

Dataframe Operations::multiply(const Dataframe& df1, const Dataframe& df2) {
    
    size_t m = df1.get_rows();
    size_t p = df2.get_cols();

    std::vector<double> res = detail::OperationsImpl::multiply_impl(
        df1.get_data(), df2.get_data(), 
        m, df1.get_cols(),
        df2.get_rows(), p,
        df1.get_storage(), df2.get_storage()
    );

    return Dataframe(m, p, false, std::move(res));
}

Dataframe Operations::transpose(Dataframe& df) {

    size_t rows = df.get_cols(), cols = df.get_rows();
    size_t temp_row = df.get_rows(), temp_col = df.get_cols();
    std::string backend = Linalg::get_backend();

    // Changing layout for better performance later
    if (df.get_storage()) df.change_layout_inplace(backend);

    std::vector<double> res = detail::OperationsImpl::transpose_impl(
        df.get_data(), temp_row, temp_col
    );

    return Dataframe(rows, cols, false, std::move(res));
}

Dataframe Operations::inverse(Dataframe& df) {
    DISPATCH_BACKEND(inverse, df)
}

std::string get_backend() {
    switch (Operations::get_backend())
    {
    case Backend::NAIVE: return "Naive";
    
    #ifdef __AVX2__
        case Backend::AVX2: return "AVX2";
        case Backend::AVX2_THREADED: return "AVX2_threaded";
    #endif

    #ifdef USE_MKL
        case Backend::MKL: return "MKL";
    #endif
    
    case Backend::EIGEN: return "Eigen"; 

    #ifdef __AVX2__
        default: return "AVX2_threaded";
    #else 
        default: return "Naive";
    #endif
    }
}

int Operations::triangular_matrix(const Dataframe& df) {

    size_t n = df.get_rows(); 
    bool is_trig_up = true;
    bool is_trig_down = true;

    // Triangular inf in row major
    for (size_t j = 0; j < n && is_trig_down; j++) {
        for(size_t i = 0; i < j && is_trig_down; i++) {
            
            if (df.at(j*n + i) != 0) is_trig_down = false;
        }
    }

    // Triangular sup in row major
    for (size_t j = 0; j < n && is_trig_up; j++) {
        for(size_t i = j+1; i < n && is_trig_up; i++) {
            
            if (df.at(j*n + i) != 0) is_trig_up = false;
        }
    }

    // We detected in row major config
    if (df.get_storage()) std::swap(is_trig_up, is_trig_down);

    if (is_trig_up && (is_trig_up && is_trig_down)) return 3; // Diag
    else if (is_trig_up) return 2; // Up
    else if (is_trig_down) return 1; // Down

    return 0; // Not triangular
}

#ifdef __AVX2__
    int Operations::triangular_matrix_avx2(const Dataframe& df) {

        size_t n = df.get_rows(); 
        bool is_trig_up = true;
        bool is_trig_down = true;

        // AVX2 variables
        size_t NB_DB = Linalg::AVX2::NB_DB;
        size_t PREFETCH_DIST = Linalg::AVX2::PREFETCH_DIST;
        __m256d zero_vec = _mm256_set1_pd(0.0);

        // Triangular inf in row major
        size_t j = 0;
        size_t vec_sizej = n - (n % NB_DB);
        for (; j < vec_sizej && is_trig_down; j+=NB_DB) {

            size_t i = 0;
            size_t vec_sizei = j - (j % NB_DB);
            for(; i < vec_sizei && is_trig_down; i+=NB_DB) {

                if (i + PREFETCH_DIST < vec_sizei) {
                    _mm_prefetch((const char*)&df.at(j*n + i + PREFETCH_DIST), _MM_HINT_T0);
                }
                __m256d vec = _mm256_loadu_pd(&df.at(j*n + i));
                __m256d cmp = _mm256_cmp_pd(vec, zero_vec, _CMP_EQ_OQ);
                int mask = _mm256_movemask_pd(cmp);
                
                if (mask != 0xF) {
                    is_trig_down = false;
                }
            }

            // Scalar residual for i
            for(; i < j && is_trig_down; i++) {
                if (df.at(j*n + i) != 0) is_trig_down = false;
            }
        }

        // Scalar residual for j
        for (; j < n && is_trig_down; j++) {
            for(size_t i = 0; i < j && is_trig_down; i++) {
                
                if (df.at(j*n + i) != 0) is_trig_down = false;
            }
        }

        // Triangular sup in row major 
        j = 0;
        for (; j < vec_sizej && is_trig_up; j+=NB_DB) {

            size_t i = j+1;
            size_t vec_sizei = n - ((n - j - 1) % NB_DB);
            for(; i < vec_sizei && is_trig_up; i+=NB_DB) {

                if (i + PREFETCH_DIST < vec_sizei) {
                    _mm_prefetch((const char*)&df.at(j*n + i + PREFETCH_DIST), _MM_HINT_T0);
                }
                __m256d vec = _mm256_loadu_pd(&df.at(j*n + i));
                __m256d cmp = _mm256_cmp_pd(vec, zero_vec, _CMP_EQ_OQ);
                int mask = _mm256_movemask_pd(cmp);
                
                if (mask != 0xF) {
                    is_trig_up = false;
                }
            }

            // Scalar residual for i
            for(;i < n && is_trig_up; i++) {
                if (df.at(j*n + i) != 0) is_trig_up = false;
            }
        }

        // Scalar residual for j
        for (; j < n && is_trig_up; j++) {
            for(size_t i = j+1; i < n && is_trig_up; i++) {
                
                if (df.at(j*n + i) != 0) is_trig_up = false;
            }
        }

        // We detected in row major config
        if (df.get_storage()) std::swap(is_trig_up, is_trig_down);

        if (is_trig_up && (is_trig_up && is_trig_down)) return 3; // Diag
        else if (is_trig_up) return 2; // Up
        else if (is_trig_down) return 1; // Down

        return 0; // Not triangular
    }
#endif

std::tuple<double, std::vector<double>, std::vector<double>> Operations::determinant(Dataframe& df) {
    
    std::string backend_str = Linalg::get_backend();

    // Changing layout for better performances
    if (df.get_storage()){
        df.change_layout_inplace(backend_str);
    }

    size_t rows = df.get_rows(), cols = df.get_cols();

    // First condition, Matrix(n,n)
    if (rows != cols) throw std::runtime_error("Need Matrix(n,n)");
    size_t n = rows;
    std::vector<double> LU;

    // Let's see if the matrix is diagonal or triangular 
    int test_v;
    if (backend_str == "Naive") test_v = triangular_matrix(df);
    #ifdef __AVX2__
        else if (backend_str == "AVX2") test_v = triangular_matrix_avx2(df);
        else test_v = triangular_matrix_avx2(df);
    #else
        else test_v = triangular_matrix(df);
    #endif

    if (test_v != 0) {
        
        double det = 1;
        for (size_t j = 0; j < n; j++) {
            
            det *= df.at(j*n + j); // Product of the diagonal
        }

        std::vector<double> vec_v = {static_cast<double>(test_v)};
        
        return std::make_tuple(det, std::move(vec_v), std::move(LU));
    }
    else {
        int nb_swaps;
        std::vector<double> swaps;

        DISPATCH_BACKEND2(LU_decomposition, df)

        double det = (nb_swaps % 2) ? -1.0 : 1.0;
        for (size_t j = 0; j < n; j++) {
            
            det *= LU[j*n + j]; // Product of the diagonal
        }
        return std::make_tuple(det, std::move(swaps), std::move(LU));
    }
}

}
