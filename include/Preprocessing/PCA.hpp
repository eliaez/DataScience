#pragma once

#include <vector>
#include <Eigen/Core>

class Dataframe;

class PCA {
    private: 
        int k_;                                     // Choosen by user
        int nb_components_;                         // K used during PCA
        std::vector<double> eigen_values_;
        std::vector<double> explained_variance_;
        std::vector<double> eigen_vectors_;         // col-major
        bool x_centered;

        // Will select the best one for you using cumulative_variance
        void select_optimal_k(const Eigen::VectorXd& cumul_variance);

        // Function to center X
        std::vector<double> center_x(const Dataframe& x) const;
        
    public:
        // k == -1 will select the best one for you using cumulative_variance
        PCA(int k = -1) : k_(k), x_centered(false) {};

        // Function for dimensionality reduction with X col major
        void fit(const Dataframe& x);

        // Function for dimensionality reduction with X col major
        Dataframe transform(const Dataframe& x);

        // Function for dimensionality reduction with X col major
        Dataframe fit_transform(const Dataframe& x);

        // Getters
        int get_nb_components() const { return nb_components_; }
        const std::vector<double>& get_eigen_values() const { return eigen_values_; }
        const std::vector<double>& get_explained_variance() const { return explained_variance_; }
        
        // Eigen vectors col-major
        const std::vector<double>& get_eigen_vectors() const { return eigen_vectors_; }
};