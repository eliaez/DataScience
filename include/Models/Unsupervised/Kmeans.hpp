#pragma once

#include <vector>
#include <string>

class Dataframe;

class Kmeans { 
    private:
        int k_;
        int n_init_; // n_init_ to perform multiple times Kmeans and then, choose the best one
        int max_iter_;
        double inertia_;
        int nb_clusters_;
        std::vector<int> labels_; // Clusters
        std::vector<std::vector<double>> cluster_centers_;

        // Function to have the best k using elbow method
        void select_optimal_k();

        // Function for the init k-means++
        std::vector<std::vector<double>> kmeans_plusplus(const std::vector<std::vector<const double*>>& X_i, const std::mt19937& rng);

        // Function to run Online Kmeans (centroid are updated after each point)
        std::pair<std::vector<int>, std::vector<double>> online_kmeans(
            const std::vector<std::vector<const double*>>& X_i, 
            std::vector<size_t>& indices,
            const std::mt19937& rng
        );

    public:
        Kmeans(int k = -1, int n_init = 10, int max_iter = 300) : k_(k),  n_init_(n_init), max_iter_(max_iter) {};

        // Kmeans function using initialisation "Kmeans++" to have a better start and results
        void fit(const Dataframe& x);

        // Kmeans function using initialisation "Kmeans++" to have a better start and results
        Dataframe predict(const Dataframe& x) const;

        // Kmeans function using initialisation "Kmeans++" to have a better start and results
        Dataframe fit_predict(const Dataframe& x);

        // Setter
        void set_n_init(int n_init) { n_init_ = n_init; }
        void set_max_iter(int max_iter) { max_iter_ = max_iter; }

        // Getters 
        int get_n_init() { return n_init_; }
        int get_max_iter() { return max_iter_; }
        double get_inertia() { return inertia_; }
        int get_nb_clusters() { return nb_clusters_; }
        const std::vector<int>& get_labels() { return labels_; }
        const std::vector<std::vector<double>>& get_cluster_centers() { return cluster_centers_; }
};