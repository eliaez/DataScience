#include <cmath>
#include <stdexcept>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Utils/ThreadPool.hpp"
#include "Preprocessing/Imputation.hpp"

using namespace Utils;

namespace Imputation {

void imputation(Dataframe& x, size_t j, const std::string& method) {
    
    if (j >= x.get_cols()) {
        throw std::invalid_argument("j >= nb_cols of x");
    }

    if (x.get_storage()) x.change_layout_inplace();

    size_t n = x.get_rows();
    if (method == "median") {
        
        // Getting data of col j, then sort it
        std::vector<double> sorted_col(n);
        for (size_t i = 0; i < n; i++) {

            // Filter NAN
            double v = x.at(j * n + i);
            if (!std::isnan(v)) sorted_col[i] = v;
        }
        std::sort(sorted_col.begin(), sorted_col.end());

        // Median for odd and even size
        double median;
        int s = static_cast<int>(sorted_col.size());
        if (s % 2 == 0) {
            median = (sorted_col[s / 2] + sorted_col[s / 2 - 1]) / 2;
        }
        else {
            median = sorted_col[(s + 1) / 2 - 1];
        }

        // Change NAN of our col
        for (size_t i = 0; i < n; i++) {
            if (std::isnan(x.at(j * n + i))) {
                x.at(j * n + i) = median;
            }   
        }
    }
    else if (method == "mode") {

        // Getting data of col j
        std::vector<double> col_j(n);
        for (size_t i = 0; i < n; i++) {
            
            // Filter NAN
            double v = x.at(j * n + i);
            if (!std::isnan(v)) col_j[i] = v;
        }

        double most = mostFrequent(col_j);

        // Change NAN of our col
        for (size_t i = 0; i < n; i++) {
            if (std::isnan(x.at(j * n + i))) {
                x.at(j * n + i) = most;
            }   
        }
    }
    else if (method == "forward") {

        // Lambda function to search for a non-NAN value until the end of the col
        auto fill_NAN = [&](size_t i) {
            size_t k = 0;
            while (std::isnan(x.at(j * n + i + 1 + k)) && (i + 1 + k < n)) {
                k++;
            }
            if (i + 1 + k >= n) {
                std::cerr << std::format("Unable to Forward fill, col of NAN from index {}\n", i);
                return std::numeric_limits<double>::quiet_NaN();
            }
            return x.at(j * n + i + 1 + k);
        };

        // Change NAN of our col
        for (size_t i = 0; i < n; i++) {
            if (std::isnan(x.at(j * n + i))) {
                x.at(j * n + i) = fill_NAN(i);
            }   
        }
    }
    else if (method == "backward") {

        // Lambda function to search for a non-NAN value until the start of the col
        auto fill_NAN = [&](size_t i) {
            size_t k = 0;
            while (std::isnan(x.at(j * n + i - 1 - k)) && (i - 1 - k > 0)) {
                k++;
            }
            if (k >= i) {
                std::cerr << std::format("Unable to Backward fill, col of NAN from index {}\n", i);
                return std::numeric_limits<double>::quiet_NaN();
            }
            return x.at(j * n + i - 1 - k);
        };

        // Change NAN of our col
        for (size_t i = n - 1; i >= 0; i--) {
            if (std::isnan(x.at(j * n + i))) {
                x.at(j * n + i) = fill_NAN(i);
            }   
        }
    }
    // Mean by default
    else if (method == "mean") {
        // Getting mean of col j
        double mean = 0.0;
        for (size_t i = 0; i < n; i++) {
            
            // Filter NAN
            double v = x.at(j * n + i);
            if (!std::isnan(v)) mean += v;
        }
        mean /= n;

        // Change NAN of our col
        for (size_t i = 0; i < n; i++) {
            if (std::isnan(x.at(j * n + i))) x.at(j * n + i) = mean;
        }
    }
    else {
        throw std::invalid_argument("Unknown method: " + method);
    }
}

void imputation(Dataframe& x, const std::string& col_name, const std::string& method) {

    // Find col
    auto headers = x.get_headers();
    auto idx = std::find(headers.begin(), headers.end(), col_name);

    if (idx != headers.end()) return imputation(x, static_cast<size_t>(idx - headers.begin()), method);
    else {
        throw std::invalid_argument(std::format("Column {} not found", col_name));
    }
}

void KNN_imputer(Dataframe& x, int K, int Lp_norm) {

    size_t n = x.get_rows();
    size_t p = x.get_cols();

    if (x.get_storage()) x.change_layout_inplace();
    Dataframe x_copy = x;

    // Getting ptrs to each row
    std::vector<std::vector<const double*>> X_i(n);
    for (size_t i = 0; i < n; i++) {
        X_i[i] = x_copy.getRowPtrs(i);
    }

    // Getting categorical cols
    std::vector<bool> col_categorical(p);
    for (size_t i = 0; i < p; i++) {
        col_categorical[i] = allIntegers(x_copy.getColumnPtrs(i));
    }

    // Only if gower distance
    std::vector<double> col_ranges;
    if (Lp_norm == -1) col_ranges = compute_ranges(x.get_data(), n, p, col_categorical, x.get_storage());

    // ThreadPool Variables
    ThreadPool& pool = ThreadPool::instance();
    size_t nb_threads = pool.nb_threads;
    std::vector<std::future<void>> futures;
    futures.reserve(nb_threads);

    // Going through the whole dataframe to detect NAN
    for (size_t j = 0; j < p; j++) {
        for (size_t i = 0; i < n; i++) {
            
            // If nan then KNN
            if (std::isnan(x.at(j * n + i))) {
                
                // Calculating distance
                std::vector<std::pair<int, double>> dist_v(n, {-1, std::numeric_limits<double>::max()});
                if (n >= 512) {
                    size_t chunk = (int)(n / nb_threads);
                    size_t start = 0, end = chunk;
                    for (size_t nb = 0; nb < nb_threads; nb++) {
                        if (nb+1 == nb_threads) end = n;

                        auto fut = pool.enqueue([start, end, i, j, n, p, Lp_norm, &x_copy, &X_i, &dist_v, &col_categorical, &col_ranges] {
                            for (size_t k = start; k < end; k++) {
                                if (std::isnan(x_copy.at(j * n + k))) continue;
                                
                                double dist;
                                if (Lp_norm == -1) dist = gower_nan(X_i[i], X_i[k], col_categorical, col_ranges);
                                else dist = Lnorm_nan(X_i[i], X_i[k], Lp_norm, 1, '-');

                                if (!std::isnan(dist)) dist_v[k] = {k, dist};
                            }
                        });
                        futures.push_back(std::move(fut));
                        start += chunk;
                        end += chunk;
                    }
                    for (auto& fut : futures) fut.wait();
                    futures.clear();
                }
                else {
                    for (size_t k = 0; k < n; k++) {

                        if (std::isnan(x_copy.at(j * n + k))) continue;
                        
                        double dist;
                        if (Lp_norm == -1) dist = gower_nan(X_i[i], X_i[k], col_categorical, col_ranges);
                        else dist = Lnorm_nan(X_i[i], X_i[k], Lp_norm, 1, '-');

                        if (!std::isnan(dist)) dist_v[k] = {k, dist};
                    }
                }

                if (dist_v.empty()) continue;

                // Sort our vector
                std::sort(dist_v.begin(), dist_v.end(), [](const auto& a, const auto& b) {
                    return a.second < b.second;
                });

                // If identical neighbor
                if (dist_v[0].second == 0.0) {
                    x.at(j * n + i) = x_copy.at(j * n + dist_v[0].first);
                }
                // categorical col
                else if (col_categorical[j]) {
                    std::unordered_map<int, double> votes;
                    size_t effective_K = std::min((size_t)K, dist_v.size());
                    for (size_t k = 0; k < effective_K; k++) {
                        int category = static_cast<int>(x_copy.at(j * n + dist_v[k].first));
                        double weight = 1.0 / dist_v[k].second;
                        votes[category] += weight;
                    }

                    // Getting best category with score
                    int best_category = -1;
                    double best_score = -1.0;
                    for (auto& [category, score] : votes) {
                        if (score > best_score) {
                            best_score = score;
                            best_category = category;
                        }
                    }

                    x.at(j * n + i) = static_cast<double>(best_category);
                }
                // Numerical col
                else {
                    double sum_num = 0.0;
                    double sum_denum = 0.0;
                    size_t effective_K = std::min((size_t)K, dist_v.size());
                    for (size_t k = 0; k < effective_K; k++) {
                        sum_num += x_copy.at(j * n + dist_v[k].first) / dist_v[k].second;
                        sum_denum += 1.0 / dist_v[k].second;
                    }
                    x.at(j * n + i) = sum_num / sum_denum;
                }
            }
        }
        // Show progression
        if (j % 2 == 0 || j == (p - 1)) {
            std::cout << "Progress: " << (j+1) << "/" << p << " (" << (100 * (j+1) / p) << "%)\n" << std::flush;
        }
    }
    std::cout << std::endl;
}
}