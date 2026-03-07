#include <numeric>
#include <stdexcept>
#include <Eigen/SVD>
#include <Eigen/Core>
#include "Data/Data.hpp"
#include "Stats/stats_reg.hpp"
#include "Preprocessing/PCA.hpp"

namespace PCA {

PCAResult PCA(const Dataframe& x, int k, bool auto_select) {

    if (x.get_storage()) {
        throw std::invalid_argument("X need to be col major");
    }

    size_t n = x.get_rows();
    size_t p = x.get_cols();

    // Centering X
    std::vector<double> x_mean(p);
    std::vector<double> x_c = x.get_data();
    for (size_t i = 0; i < p; i++) {
        
        // Pointer to the start of column i
        double* col_start = x_c.data() + i * n;
        
        // Mean
        x_mean[i] = std::accumulate(col_start, col_start + n, 0.0) / n;
        
        // Center
        for (size_t j = 0; j < n; j++) {
            col_start[j] -= x_mean[i];
        }
    }

    // SVD on X_centered
    Eigen::Map<const Eigen::MatrixXd> X_c(x_c.data(), n, p);
    Eigen::BDCSVD<Eigen::MatrixXd> svd(X_c, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Getting eigen values
    Eigen::VectorXd singular_values = svd.singularValues();
    Eigen::VectorXd eigen_v = singular_values.array().square() / n;
    
    double total = eigen_v.sum();
    Eigen::VectorXd explained_v = eigen_v / total;

    // Cumulated Variance
    Eigen::VectorXd cumulative_variance = Eigen::VectorXd::Zero(explained_v.size());
    cumulative_variance(0) = explained_v(0);
    for (int i = 1; i < explained_v.size(); i++) {
        cumulative_variance(i) = cumulative_variance(i-1) + explained_v(i);
    }

    // Calculate optimal k
    int opt_k = 1;
    for (int i = 0; i < cumulative_variance.size(); i++) {
        if (cumulative_variance(i) >= 0.95) {
            opt_k = i + 1;
            break;
        }
    }

    // Project 
    Eigen::MatrixXd Wk = auto_select ? svd.matrixV().leftCols(opt_k) : svd.matrixV().leftCols(k);    
    Eigen::MatrixXd Z = X_c * Wk;
    std::vector<double> proj(Z.data(), Z.data() + Z.size());

    // Getting results
    PCAResult res;
    res.k = auto_select ? opt_k : k;
    res.optimal_k = opt_k;
    res.projection = Dataframe(n, res.k, false, proj);
    res.eigenvalues.assign(eigen_v.data(), eigen_v.data() + eigen_v.size());
    res.explained_variance.assign(explained_v.data(), explained_v.data() + explained_v.size());

    return res;
}
}

