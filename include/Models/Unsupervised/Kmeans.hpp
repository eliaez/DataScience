#pragma once

#include <vector>
#include <string>
#include <random>

class Dataframe;

class Kmeans { 
    private:
        int k_;
        int n_init_; // n_init_ to perform multiple times Kmeans and then, choose the best one
        int max_iter_;
        double inertia_;
        std::string method_;
        std::vector<int> labels_; // Clusters
        std::pair<int, int> min_max_auto; // Min max for select_optimal_k if k == -1 
        std::vector<std::vector<double>> cluster_centers_;

        // Function to have the best k using elbow method
        void select_optimal_k(const Dataframe& x);

        // Function for the init k-means++
        std::vector<std::vector<double>> kmeans_plusplus(const std::vector<std::vector<const double*>>& X_i, std::mt19937& rng);

        // Function to run Online Kmeans (centroid are updated after each point)
        std::pair<std::vector<int>, std::vector<std::vector<double>>> online_kmeans(
            const std::vector<std::vector<const double*>>& X_i, 
            std::vector<size_t>& indices,
            std::mt19937& rng
        );

        // Function to run Batch Kmeans (default one) (centroid are updated after N points)
        std::pair<std::vector<int>, std::vector<std::vector<double>>> batch_kmeans(
            const std::vector<std::vector<const double*>>& X_i,
            std::mt19937& rng
        );

        // Function to run MiniBatch Kmeans (default one) (centroid are updated after sqrt(N) points)
        std::pair<std::vector<int>, std::vector<std::vector<double>>> minibatch_kmeans(
            const std::vector<std::vector<const double*>>& X_i, 
            std::vector<size_t>& indices,
            std::mt19937& rng
        );

    public:
        // If k == -1, algo will select for you the optimal nb of clusters using elbow method
        // n_init for nb of runs and will keep the best one (10 by default), 
        // max_iter is a condition if threshold of 1e-4 for convergence isn't enough (300 by default),
        // method are "kmeans" (by default classical one), "minibatch", "online"
        Kmeans(int k = -1, int n_init = 10, int max_iter = 300, std::string method = "kmeans", std::pair<int, int> min_max_ = {2, 10}) : 
        k_(k),  n_init_(n_init), max_iter_(max_iter), method_(method), min_max_auto(min_max_) {};

        // Kmeans function using initialisation "Kmeans++" to have a better start and results
        std::vector<int> predict(const Dataframe& x);

        // Kmeans function using initialisation "Kmeans++" to have a better start and results
        std::vector<int> fit_predict(const Dataframe& x, bool show_progression = true);

        // Setter
        void set_n_init(int n_init) { n_init_ = n_init; }
        void set_max_iter(int max_iter) { max_iter_ = max_iter; }
        void set_minmax(std::pair<int, int> min_max_) { min_max_auto = min_max_; }

        // Getters 
        int get_n_init() { return n_init_; }
        int get_max_iter() { return max_iter_; }
        double get_inertia() { return inertia_; }
        int get_nb_clusters() { return k_; }
        std::pair<int, int> get_minmax() { return min_max_auto; }
        const std::vector<int>& get_labels() { return labels_; }
        const std::vector<std::vector<double>>& get_cluster_centers() { return cluster_centers_; }
};