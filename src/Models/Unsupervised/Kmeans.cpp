#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Models/Unsupervised/Kmeans.hpp"

using namespace Utils;

void Kmeans::select_optimal_k(const Dataframe& x) {

    // Runnning for each k
    std::vector<double> inertia_v;
    for (size_t i = min_max_auto.first; i < min_max_auto.second; i++) {
        k_ = i;
        fit_predict(x, false);
        inertia_v.push_back(inertia_);

        if (i % 1 == 0 || i == (min_max_auto.second - 1)) {
            std::cout << "Progress to select optimal k: " << (i+1-min_max_auto.first) << "/" << (min_max_auto.second-min_max_auto.first) << " (" << (100 * (i+1-min_max_auto.first) / (min_max_auto.second-min_max_auto.first)) << "%)\n" << std::flush;
        }   
    }

    // Getting best one
    std::vector<double> diff(inertia_v.size() - 1); 
    for (size_t i = 0; i< inertia_v.size() - 1; i++) 
        diff[i] =  inertia_v[i + 1] - inertia_v[i];

    std::vector<double> diff1(diff.size() - 1); 
    for (size_t i = 0; i< diff.size() - 1; i++) 
        diff1[i] =  diff[i + 1] - diff[i];

    auto it = std::max_element(diff1.begin(), diff1.end());
    k_ = min_max_auto.first + std::distance(diff1.begin(), it) + 2;
}

std::vector<int> Kmeans::predict(const Dataframe& x) {

    size_t n = x.get_rows();
    
    // Getting ptrs to each row
    std::vector<std::vector<const double*>> X_i(n);
    for (size_t i = 0; i < n; i++) {
        X_i[i] = x.getRowPtrs(i);
    }
    
    labels_.resize(n);
    for (size_t i = 0; i < n; i++) {

        // Assigning a cluster to obs i 
        std::vector<double> center_dist(k_);
        for (size_t j = 0; j < k_; j++) 
            center_dist[j] = Lnorm(X_i[i], cluster_centers_[j], 2, 1, '-');
        
        labels_[i] = std::min_element(center_dist.begin(), center_dist.end()) - center_dist.begin();
    }
    
    return labels_;
}

std::vector<int> Kmeans::fit_predict(const Dataframe& x, bool show_progression) {
    size_t n = x.get_rows();

    if (k_ == -1) select_optimal_k(x);

    // Vector of indices
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // For random shuffle later
    std::mt19937 rng;
    rng.seed(std::random_device{}());

    // Getting ptrs to each row
    std::vector<std::vector<const double*>> X_i(n);
    for (size_t i = 0; i < n; i++) {
        X_i[i] = x.getRowPtrs(i);
    }

    // Doing n_init kmeans to keep best one
    std::vector<std::vector<int>> all_labels;
    std::vector<double> inertia_v(n_init_, 0.0);
    std::vector<std::vector<std::vector<double>>> all_clusters;
    all_labels.reserve(n_init_);
    all_clusters.reserve(n_init_);
    for (size_t k = 0; k < n_init_; k++) {

        // Choosing according to method variable
        std::vector<int> labels;
        std::vector<std::vector<double>> clusters;
        if (method_ == "kmeans") {
            std::tie(labels, clusters) = batch_kmeans(X_i, rng);
        }
        else if (method_ == "minibatch") {
            std::tie(labels, clusters) = minibatch_kmeans(X_i, indices, rng);
        }
        else if (method_ == "online") {
            std::tie(labels, clusters) = online_kmeans(X_i, indices, rng);
        }
        else throw std::invalid_argument("Unknown method: " + method_);

        // Calculate inertia
        for (size_t i = 0; i < n; i++) {
            inertia_v[k] += Lnorm(X_i[i], clusters[labels[i]], 2, 2, '-');
        }
        all_labels.push_back(std::move(labels));
        all_clusters.push_back(std::move(clusters));

        // Show progression
        if (show_progression) {
            if (k % 1 == 0 || k == (n_init_ - 1)) {
                std::cout << "Progress: " << (k+1) << "/" << n_init_ << " (" << (100 * (k+1) / n_init_) << "%)\n" << std::flush;
            }
        }    
    }
    std::cout << std::endl;

    // Keeping the best one
    auto it = std::min_element(inertia_v.begin(), inertia_v.end());
    inertia_ = *it; 

    // Getting result
    labels_ = all_labels[it - inertia_v.begin()];
    cluster_centers_ = all_clusters[it - inertia_v.begin()];
    return labels_;
}

std::vector<std::vector<double>> Kmeans::kmeans_plusplus(const std::vector<std::vector<const double*>>& X_i, std::mt19937& rng) {
    
    size_t n = X_i.size();
    size_t p = X_i[0].size();

    // Getting k centroids with kmeans++
    std::vector<std::vector<double>> D(n);
    std::vector<std::vector<double>> clusters;
    std::uniform_int_distribution<int> dist(0, n - 1);
    clusters.reserve(k_);
    for (size_t i = 0; i < k_; i++) {

        size_t chosen;
        if (i == 0) 
            chosen = dist(rng);  // First centroid
        else {
            // Getting min dist for each obs to each centroid
            std::vector<double> min_D;
            min_D.reserve(n);
            for (size_t j = 0; j < n; j++) {
                D[j].push_back(Lnorm(X_i[j], clusters[i-1], 2, 2, '-'));

                auto it = std::min_element(D[j].begin(), D[j].end()) - D[j].begin();
                min_D.push_back(D[j][it]);
            }

            // Build cumulative distribution function (CDF) from squared distances
            std::vector<double> cdf(n);
            cdf[0] = min_D[0];
            for (int j = 1; j < n; j++)
                cdf[j] = cdf[j-1] + min_D[j];

            // Draw a random value in [0, total_sum]
            std::uniform_real_distribution<double> real_dist(0.0, cdf[n-1]);
            double r = real_dist(rng);

            // Find the corresponding point
            chosen = lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin();
        }

        // Assigning chosen point as new cluster
        std::vector<double> chosen_point(p);
        for (size_t l = 0; l < p; l++) {
            chosen_point[l] = (*X_i[chosen][l]);
        }
        clusters.push_back(std::move(chosen_point));
    }
    return clusters;
}

std::pair<std::vector<int>, std::vector<std::vector<double>>> Kmeans::online_kmeans(
    const std::vector<std::vector<const double*>>& X_i, std::vector<size_t>& indices, std::mt19937& rng) {
    
    // Init
    size_t n = X_i.size();
    size_t p = X_i[0].size();
    std::vector<std::vector<double>> clusters = kmeans_plusplus(X_i, rng);

    // Until cv
    size_t idx = 0;
    bool keep_cond = true;
    std::vector<int> labels(n, -1);
    std::vector<int> count_clusters(k_);
    std::vector<std::vector<double>> old_clusters(k_);
    while (keep_cond && idx < max_iter_) {

        // Reset count and update old_clusters
        old_clusters = clusters;
        std::fill(count_clusters.begin(), count_clusters.end(), 0);
        
        // For each obs calculate it's euclidian distance from each centroid
        std::shuffle(indices.begin(), indices.end(), rng);
        for (size_t i : indices) {

            // Assigning a cluster to obs i 
            std::vector<double> center_dist(k_);
            for (size_t j = 0; j < k_; j++) 
                center_dist[j] = Lnorm(X_i[i], clusters[j], 2, 2, '-');

            labels[i] = std::min_element(center_dist.begin(), center_dist.end()) - center_dist.begin();

            // Updating cluster
            count_clusters[labels[i]]++;
            for (size_t j = 0; j < p; j++)
                clusters[labels[i]][j] += (*X_i[i][j] - clusters[labels[i]][j]) / count_clusters[labels[i]];
        }

        // Check Cv
        if (idx >= 1) {
            keep_cond = false;
            for (size_t j = 0; j < k_; j++) {
                if (Lnorm(clusters[j], old_clusters[j], 2, 2, '-') > 1e-4) {
                    keep_cond = true;
                    break;
                }
            }
        }
        idx++;
    }
    return {labels, clusters};
}

std::pair<std::vector<int>, std::vector<std::vector<double>>> Kmeans::batch_kmeans(
    const std::vector<std::vector<const double*>>& X_i, std::mt19937& rng) {
    
    // Init
    size_t n = X_i.size();
    size_t p = X_i[0].size();
    std::vector<std::vector<double>> clusters = kmeans_plusplus(X_i, rng);

    // Until cv
    size_t idx = 0;
    bool keep_cond = true;
    std::vector<int> labels(n, -1);
    std::vector<int> count_clusters(k_);
    std::vector<std::vector<double>> old_clusters(k_);
    while (keep_cond && idx < max_iter_) {

        // Reset new and update old clusters
        old_clusters = clusters;
        std::vector<std::vector<double>> new_clusters(k_, std::vector<double>(p, 0.0));
        std::fill(count_clusters.begin(), count_clusters.end(), 0);
        
        // For each obs calculate it's euclidian distance from each centroid
        for (size_t i = 0; i < n; i++) {

            // Assigning a cluster to obs i 
            std::vector<double> center_dist(k_);
            for (size_t j = 0; j < k_; j++) 
                center_dist[j] = Lnorm(X_i[i], old_clusters[j], 2, 2, '-');

            labels[i] = std::min_element(center_dist.begin(), center_dist.end()) - center_dist.begin();
        }

        // Updating clusters
        for (size_t i = 0; i < n; i++) {
            
            count_clusters[labels[i]]++;
            for (size_t j = 0; j < p; j++)
                new_clusters[labels[i]][j] += (*X_i[i][j] - new_clusters[labels[i]][j]) / count_clusters[labels[i]];
        }
        clusters = new_clusters;

        // Check Cv
        if (idx >= 1) {
            keep_cond = false;
            for (size_t j = 0; j < k_; j++) {
                if (Lnorm(clusters[j], old_clusters[j], 2, 2, '-') > 1e-4) {
                    keep_cond = true;
                    break;
                }
            }
        }
        idx++;
    }
    return {labels, clusters};
}

std::pair<std::vector<int>, std::vector<std::vector<double>>> Kmeans::minibatch_kmeans(
    const std::vector<std::vector<const double*>>& X_i, std::vector<size_t>& indices, std::mt19937& rng) {
    
    // Init
    size_t n = X_i.size();
    size_t p = X_i[0].size();
    std::vector<std::vector<double>> clusters = kmeans_plusplus(X_i, rng);

    // Until cv
    size_t idx = 0;
    bool keep_cond = true;
    std::vector<int> labels(n, -1);
    std::vector<int> count_clusters(k_);
    std::vector<std::vector<double>> old_clusters(k_);
    while (keep_cond && idx < max_iter_) {

        // Reset count and update old clusters
        old_clusters = clusters;
        std::fill(count_clusters.begin(), count_clusters.end(), 0);
        
        // For each obs calculate it's euclidian distance from each centroid
        std::shuffle(indices.begin(), indices.end(), rng);
        std::vector<size_t> id(indices.begin(), indices.begin() + static_cast<size_t>(std::sqrt(n)));
        for (size_t i : id) {

            // Assigning a cluster to obs i 
            std::vector<double> center_dist(k_);
            for (size_t j = 0; j < k_; j++) 
                center_dist[j] = Lnorm(X_i[i], clusters[j], 2, 2, '-');

            labels[i] = std::min_element(center_dist.begin(), center_dist.end()) - center_dist.begin();
        }

        // Updating clusters
        for (size_t i : id) {
            count_clusters[labels[i]]++;
            for (size_t j = 0; j < p; j++)
                clusters[labels[i]][j] += (*X_i[i][j] - clusters[labels[i]][j]) / count_clusters[labels[i]];
        }

        // Check Cv
        if (idx >= 1) {
            keep_cond = false;
            for (size_t j = 0; j < k_; j++) {
                if (Lnorm(clusters[j], old_clusters[j], 2, 2, '-') > 1e-4) {
                    keep_cond = true;
                    break;
                }
            }
        }
        idx++;
    }

    // Final labeling for all points
    for (size_t i = 0; i < n; i++) {

        std::vector<double> center_dist(k_);
        for (size_t j = 0; j < k_; j++) {
            center_dist[j] = Lnorm(X_i[i], clusters[j], 2, 2, '-');
        }
        labels[i] = std::min_element(center_dist.begin(), center_dist.end()) - center_dist.begin();
    }
    return {labels, clusters};
}