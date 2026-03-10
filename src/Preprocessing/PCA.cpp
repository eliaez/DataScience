#include <numeric>
#include <stdexcept>
#include <Eigen/SVD>
#include "Data/Data.hpp"
#include "Stats/stats_reg.hpp"
#include "Preprocessing/PCA.hpp"

void PCA::select_optimal_k(const Eigen::VectorXd& cumul_variance) {
    // Calculate optimal k
    int opt_k = 1;
    for (int i = 0; i < cumul_variance.size(); i++) {
        if (cumul_variance(i) >= 0.95) {
            opt_k = i + 1;
            break;
        }
    }
    nb_components_ = opt_k;
}

void PCA::fit(const Dataframe& x) {
        if (x.get_storage()) {
        throw std::invalid_argument("X need to be col major");
    }

    size_t n = x.get_rows();
    size_t p = x.get_cols();

    // Centering X
    std::vector<double> x_mean(p);
    std::vector<double> x_c = center_x(x);

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
    for (int i = 1; i < explained_v.size(); i++) 
        cumulative_variance(i) = cumulative_variance(i-1) + explained_v(i);

    // Optimal k
    if (k_ == -1) select_optimal_k(cumulative_variance);
    else nb_components_ = k_;
    
    // Getting results   
    eigen_values_.assign(eigen_v.data(), eigen_v.data() + eigen_v.size());
    explained_variance_.assign(explained_v.data(), explained_v.data() + explained_v.size());

    // Getting eigen_vector
    Eigen::MatrixXd Wk = svd.matrixV().leftCols(nb_components_); 
    eigen_vectors_.assign(Wk.data(), Wk.data() + Wk.size());
}

Dataframe PCA::transform(const Dataframe& x) {

    if (x.get_storage()) {
        throw std::invalid_argument("X need to be col major");
    }

    size_t n = x.get_rows();
    size_t p = x.get_cols();

    // Centering X
    std::vector<double> x_mean(p);
    std::vector<double> x_c = x_centered ? center_x(x) : x.get_data();
    if (x_centered) x_centered = false;

    // Projection
    Eigen::Map<const Eigen::MatrixXd> X_c(x_c.data(), n, p);
    Eigen::Map<const Eigen::MatrixXd> Wk(eigen_vectors_.data(), p, nb_components_);

    Eigen::MatrixXd Z = X_c * Wk;
    std::vector<double> proj(Z.data(), Z.data() + Z.size());

    // Getting results
    Dataframe res = Dataframe(n, nb_components_, false, proj);

    return res;
}

Dataframe PCA::fit_transform(const Dataframe& x) {
    fit(x);
    x_centered = true;
    return transform(x);
}

std::vector<double> PCA::center_x(const Dataframe& x) const {
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
        for (size_t j = 0; j < n; j++) 
            col_start[j] -= x_mean[i];
    }
    return x_c;
}

