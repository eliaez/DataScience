#include "Models/Regressions.hpp"
#include "Linalg/detail/LinalgImpl.hpp"
#include "Stats/stats.hpp"

#include <stdexcept>

namespace Reg {

void fit(const std::vector<double>& x, const std::vector<double>& y) {

    size_t n = x.size();

    // Insert an unit col to get intercept value
    std::vector<double> X = x;
    for (size_t i = 0; i < n; i++) {
        X.insert(X.begin(), 1.0);
    }

    //auto X_t = Linalg::detail::OperationsImpl::transpose_impl(X, n*2, n*2, false); 
    
    // Calculate Beta (our estimator)
    //std::vector<double> beta_est =  
    
}


}