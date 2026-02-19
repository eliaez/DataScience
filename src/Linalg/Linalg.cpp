#include "Data/Data.hpp"
#include "Linalg/Linalg.hpp"
#include "Linalg/detail/LinalgImpl.hpp"

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

    std::vector<double> res = Impl::sum_impl(
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

    std::vector<double> res = Impl::multiply_impl(
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

    std::vector<double> res = Impl::transpose_impl(
        df.get_data(), temp_row, temp_col, df.get_storage()
    );

    return Dataframe(rows, cols, false, std::move(res));
}

Dataframe Operations::inverse(Dataframe& df) {
    
    size_t m = df.get_rows();
    size_t n = df.get_cols();
    std::string backend = Linalg::get_backend();

    // Verify if we have a square matrix
    if (m != n) throw std::runtime_error("Need Matrix(n,n)");

    // Changing layout for better performance later
    if (df.get_storage()) df.change_layout_inplace(backend);

    std::vector<double> res = Impl::inverse_impl(
        df.get_data(), n, df.get_storage()
    );

    return Dataframe(n, n, false, std::move(res));
}

std::tuple<double, std::vector<double>, std::vector<double>> Operations::determinant(Dataframe& df) {

    size_t m = df.get_rows();
    size_t n = df.get_cols();
    std::string backend = Linalg::get_backend();

    // Verify if we have a square matrix
    if (m != n) throw std::runtime_error("Need Matrix(n,n)");

    // Changing layout for better performance later
    if (df.get_storage()) df.change_layout_inplace(backend);

    return Impl::determinant_impl(
        df.get_data(), n, df.get_storage()
    );
}

int Operations::triangular_matrix(const Dataframe& df) {

    return Impl::triangular_impl(
        df.get_data(), df.get_rows(), df.get_cols(), df.get_storage()
    );
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
}
