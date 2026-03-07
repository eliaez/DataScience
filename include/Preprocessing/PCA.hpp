#pragma once

class Dataframe;

namespace PCA {

    struct PCAResult {
        int k;
        int optimal_k;
        Dataframe projection;
        std::vector<double> eigenvalues;
        std::vector<double> explained_variance;
    };

    // Function for dimensionality reduction with X col major
    // Auto select bool will select the optimal k for you using cumulative_variance
    PCAResult PCA(const Dataframe& x, int k, bool auto_select = true);
}