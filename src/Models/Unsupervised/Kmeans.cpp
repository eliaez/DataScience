#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Models/Unsupervised/Kmeans.hpp"

using namespace Utils;

void Kmeans::select_optimal_k() {

    std::vector<double> diff(inertia_.size() - 1); 
    for (size_t i = 0; i< inertia_.size() - 1; i++) 
        diff[i] =  inertia_[i + 1] - inertia_[i];

    std::vector<double> diff1(diff.size() - 1); 
    for (size_t i = 0; i< diff.size() - 1; i++) 
        diff1[i] =  diff[i + 1] - diff[i];

    auto it = std::max_element(diff1.begin(), diff1.end());
    nb_clusters_ = std::distance(diff1.begin(), it) + 2;
}

std::vector<std::vector<double>> Kmeans::kmeans_plusplus(const std::vector<std::vector<const double*>>& X_i, const std::mt19937& rng) {
    
    size_t n = X_i.size();
    size_t p = X_i[0].size();

    // Getting k centroids with kmeans++
    std::vector<std::vector<double>> D(n);
    std::vector<std::vector<double>> clusters;
    std::uniform_int_distribution<int> dist(0, n - 1);
    clusters.reserve(k_);
    for (size_t i = 0; i < k_; i++) {

        int chosen;
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
            double r = (static_cast<double>(rand()) / RAND_MAX) * cdf[n-1];

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
}

void Kmeans::fit(const Dataframe& x) {
    size_t n = x.get_rows();
    size_t p = x.get_cols();

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

    std::vector<std::vector<int>> all_labels;
    std::vector<double> inertia_v(n_init_, 0.0);
    std::vector<std::vector<std::vector<double>>> all_clusters;
    all_labels.reserve(n_init_);
    all_clusters.reserve(n_init_);
    for (size_t k = 0; k < n_init_; k++) {

        auto [labels, clusters] = online_kmeans(X_i, indices, rng);

        // Calculate inertia
        for (size_t i = 0; i < n; i++) {
            inertia_v[k] += Lnorm(X_i[i], clusters[labels[i]], 2, 2, '-');
        }
        all_labels.push_back(std::move(labels));
        all_clusters.push_back(std::move(clusters));
    }

    // Keeping the best one
    auto it = std::min_element(inertia_v.begin(), inertia_v.end());
    inertia_ = *it; 

    // Getting result
    labels_ = all_labels[it - inertia_v.begin()];
    cluster_centers_ = all_clusters[it - inertia_v.begin()];
}

std::pair<std::vector<int>, std::vector<std::vector<double>>> Kmeans::online_kmeans(
    const std::vector<std::vector<const double*>>& X_i, std::vector<size_t>& indices, const std::mt19937& rng) {
    
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
        
        // For each obs calculate it's euclidian distance from each centroid
        std::shuffle(indices.begin(), indices.end(), rng);
        for (size_t i : indices) {

            // Assigning a cluster to obs i 
            std::vector<double> center_dist(k_);
            for (size_t j = 0; j < k_; j++) 
                center_dist[j] = Lnorm(X_i[i], clusters[j], 2, 1, '-');

            int it = std::min_element(center_dist.begin(), center_dist.end()) - center_dist.begin();
            int old_idx = labels[i];
            labels[i] = it;

            // Updating cluster
            count_clusters[labels[i]]++;
            for (size_t j = 0; j < p; j++)
                clusters[labels[i]][j] += (*X_i[i][j] - clusters[labels[i]][j]) / count_clusters[labels[i]];

            // If he was in an another cluster
            if (old_idx != -1 && count_clusters[old_idx] > 0) {
                count_clusters[old_idx]--;
                if (count_clusters[old_idx] > 0)
                    for (size_t j = 0; j < p; j++)
                        clusters[old_idx][j] = (clusters[old_idx][j] * (count_clusters[old_idx] + 1) - *X_i[i][j]) / count_clusters[old_idx];
            }
        }

        // Check Cv
        if (idx >= 1) {
            keep_cond = false;
            for (size_t j = 0; j < k_; j++) {
                if (Lnorm(clusters[j], old_clusters[j], 2, 1, '-') > 1e-4) {
                    keep_cond = true;
                    break;
                }
            }
        }
        old_clusters = clusters;
        idx++;
    }
    return {labels, clusters};
}