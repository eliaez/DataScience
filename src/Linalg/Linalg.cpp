#include "Linalg/Linalg.hpp"

/*case Linalg::Backend::AVX2: return Linalg::Avx2::func(__VA_ARGS__); \
        case Linalg::Backend::AVX2_THREADED: return Linalg::Avx2_threaded::func(__VA_ARGS__); \
        case Linalg::Backend::EIGEN: return Linalg::Eigen::func(__VA_ARGS__); \ 
*/

#define DISPATCH_BACKEND(func, ...) \
    switch(current_backend) { \
        case Linalg::Backend::NAIVE: return Linalg::Naive::func(__VA_ARGS__); \
        default: return Linalg::Naive::func(__VA_ARGS__); \
    }

namespace Linalg {

Backend Operations::current_backend = Backend::AUTO;

void Operations::set_backend(Backend b) {
    current_backend = b;
}

Backend Operations::get_backend() {
    
    // Select the best one
    if (current_backend == Backend::AUTO) {

        // Naive for now
        return Backend::NAIVE;

        // return Backend::AVX2_THREADED;
    } 
    return current_backend;
}

Dataframe Operations::sum(const Dataframe& df1, const Dataframe& df2, char op) {
    DISPATCH_BACKEND(sum, df1, df2, op)
}

Dataframe Operations::multiply(const Dataframe& df1, Dataframe& df2) {
    DISPATCH_BACKEND(multiply, df1, df2)
}

Dataframe Operations::transpose(Dataframe& df) {
    DISPATCH_BACKEND(transpose, df)
}

Dataframe Operations::inverse(Dataframe& df) {
    DISPATCH_BACKEND(inverse, df)
}


}
