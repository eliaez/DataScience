#include "Linalg/Linalg.hpp"

#if defined(__AVX2__) && defined(USE_MKL)
    #define DISPATCH_BACKEND(func, ...) \
        switch(current_backend) { \
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
        switch(current_backend) { \
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
        switch(current_backend) { \
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
        switch(current_backend) { \
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
    DISPATCH_BACKEND(sum, df1, df2, op)
}

Dataframe Operations::multiply(const Dataframe& df1, const Dataframe& df2) {
    DISPATCH_BACKEND(multiply, df1, df2)
}

Dataframe Operations::transpose(Dataframe& df) {
    DISPATCH_BACKEND(transpose, df)
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

int triangular_matrix(const Dataframe& df) {

    size_t n = df.get_rows(); 
    bool is_trig_up = true;
    bool is_trig_down = true;

    // Triangular sup
    for (size_t i = 0; i < n && is_trig_up; i++) {
        for(size_t j = 0; j < i && is_trig_up; j++) {
            
            if (df.at(j*n + i) != 0) is_trig_up = false;
        }
    }

    // Triangular inf 
    for (size_t i = 0; i < n && is_trig_down; i++) {
        for(size_t j = i+1; j < n && is_trig_down; j++) {
            
            if (df.at(j*n + i) != 0) is_trig_down = false;
        }
    }

    if (is_trig_up && (is_trig_up && is_trig_down)) return 3; // Diag
    else if (is_trig_up) return 2; // Up
    else if (is_trig_down) return 1; // Down

    return 0; // Not triangular
}

std::tuple<double, std::vector<double>, std::vector<double>> determinant(Dataframe& df) {
    
    // Changing layout for better performances
    if (df.get_storage()){
        df.change_layout_inplace(get_backend());
    }

    size_t rows = df.get_rows(), cols = df.get_cols();

    // First condition, Matrix(n,n)
    if (rows != cols) throw std::runtime_error("Need Matrix(n,n)");
    size_t n = rows;
    std::vector<double> LU;

    // Let's see if the matrix is diagonal or triangular 
    int test_v = triangular_matrix(df);
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
