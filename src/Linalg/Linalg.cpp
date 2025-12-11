#include "Linalg/Linalg.hpp"

/*
        case Linalg::Backend::AVX2_THREADED: return Linalg::Avx2_threaded::func(__VA_ARGS__); \
        case Linalg::Backend::EIGEN: return Linalg::Eigen::func(__VA_ARGS__); \ 
*/

#ifdef __AVX2__
    #define DISPATCH_BACKEND(func, ...) \
        switch(current_backend) { \
            case Linalg::Backend::NAIVE: return Linalg::Naive::func(__VA_ARGS__); \
            case Linalg::Backend::AVX2: return Linalg::AVX2::func(__VA_ARGS__); \
            default: return Linalg::AVX2::func(__VA_ARGS__); \
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
            default: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::AVX2::func(__VA_ARGS__); \
                break; \
            } \
        } 
#else
    #define DISPATCH_BACKEND(func, ...) \
        switch(current_backend) { \
            case Linalg::Backend::NAIVE: return Linalg::Naive::func(__VA_ARGS__); \
            case Linalg::Backend::EIGEN: return Linalg::Eigen::func(__VA_ARGS__); \ 
            default: return Linalg::NAIVE::func(__VA_ARGS__); \
        }

    #define DISPATCH_BACKEND2(func, ...) \
        switch(Operations::get_backend()) { \
            case Linalg::Backend::NAIVE: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::Naive::func(__VA_ARGS__); \
                break; \
            } \
            case Linalg::Backend::EIGEN: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::EIGEN::func(__VA_ARGS__); \
                break; \
            } \
            default: { \
                std::tie(nb_swaps, swaps, LU) = Linalg::NAIVE::func(__VA_ARGS__); \
                break; \
            } \
        } 
#endif

namespace Linalg {

Backend Operations::current_backend = Backend::AUTO;

void Operations::set_backend(Backend b) {
    current_backend = b;
}

Backend Operations::get_backend() {
    
    // Select the best one
    if (current_backend == Backend::AUTO) {

        // AVX2 for now
        #ifdef __AVX2__
            return Backend::AVX2;
        #else
            return Backend::NAIVE;
        #endif

        // return Backend::AVX2_THREADED;
    } 
    return current_backend;
}

Dataframe Operations::sum(Dataframe& df1, Dataframe& df2, char op) {
    DISPATCH_BACKEND(sum, df1, df2, op)
}

Dataframe Operations::multiply(Dataframe& df1, Dataframe& df2) {
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
    
    case Backend::EIGEN: return "Eigen"; 

    #ifdef __AVX2__
        default: return "AVX2";
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

std::tuple<double, std::vector<double>, Dataframe>determinant(Dataframe& df) {
    
    // Changing layout for better performances
    if (df.get_storage()){
        df.change_layout_inplace(get_backend());
    }

    size_t rows = df.get_rows(), cols = df.get_cols();

    // First condition, Matrix(n,n)
    if (rows != cols) throw std::runtime_error("Need Matrix(n,n)");
    size_t n = rows;

    // Let's see if the matrix is diagonal or triangular 
    int test_v = triangular_matrix(df);
    if (test_v != 0) {
        
        double det = 1;
        for (size_t j = 0; j < n; j++) {
            
            det *= df.at(j*n + j); // Product of the diagonal
        }
        return {det, {static_cast<double>(test_v)}, {}};
    }
    else {
        int nb_swaps;
        std::vector<double> swaps;
        Dataframe LU; 

        DISPATCH_BACKEND2(LU_decomposition, df)

        double det = (nb_swaps % 2) ? -1.0 : 1.0;
        for (size_t j = 0; j < n; j++) {
            
            det *= LU.at(j*n + j); // Product of the diagonal
        }
        return {det, swaps, LU};
    }
}

}
