#include <cmath>
#include <stdexcept>
#include "Data/Data.hpp"
#include "Stats/stats_reg.hpp"
#include "Preprocessing/Scaling.hpp"

namespace Scaling {

void scaling(Dataframe& x, size_t j, const std::string& method, double min, double max) {

    if (j >= x.get_cols()) {
        throw std::invalid_argument("j >= nb_cols of x");
    }

    if (x.get_storage()) x.change_layout_inplace();

    size_t n = x.get_rows();
    if (method == "percentile") {
        
        // Getting data of col j, then sort it
        std::vector<double> sorted_col(n);
        for (size_t i = 0; i < n; i++) {
            sorted_col[i] = x.at(j * n + i);
        }
        std::sort(sorted_col.begin(), sorted_col.end());

        // Lambda function to calc quantiles
        auto calc_quantile = [&](double q) {
            double pos = q * (n - 1);
            int idx = static_cast<int>(pos);
            double frac = pos - idx;
            return (idx + 1 < n) ? 
                sorted_col[idx] + frac * (sorted_col[idx + 1] - sorted_col[idx]) : sorted_col[idx];
        };

        // Quantiles
        double Q1 = calc_quantile(0.25);
        double Q2 = calc_quantile(0.5);
        double Q3 = calc_quantile(0.75);

        double test = Q3 - Q1;
        if (test == 0.0) throw std::runtime_error("Scaling impossible: Q3 = Q1");

        // Change data of our col
        for (size_t i = 0; i < n; i++) {
            x.at(j * n + i) = (x.at(j * n + i) - Q2) / (Q3 - Q1);
        }
    }
    else if (method == "mean") {

        // Getting mean, max, min of col j
        double mean = 0.0;
        double min_x = x.at(j * n + 0);
        double max_x = x.at(j * n + 0);
        for (size_t i = 0; i < n; i++) {
            
            if (x.at(j * n + i) < min_x) {
                min_x = x.at(j * n + i);
            }
            if (x.at(j * n + i) > max_x) {
                max_x = x.at(j * n + i);
            }
            mean += x.at(j * n + i);
        }
        mean /= n;

        double test = max_x - min_x;
        if (test == 0.0) throw std::runtime_error("Scaling impossible: max col = min col");

        // Change data of our col
        for (size_t i = 0; i < n; i++) {
            x.at(j * n + i) = (x.at(j * n + i) - mean) / (max_x - min_x);
        }
    }
    else if (method == "minmax") {

        // Getting max, min of col j
        double min_x = x.at(j * n + 0);
        double max_x = x.at(j * n + 0);
        for (size_t i = 0; i < n; i++) {
            
            if (x.at(j * n + i) < min_x) {
                min_x = x.at(j * n + i);
            }
            if (x.at(j * n + i) > max_x) {
                max_x = x.at(j * n + i);
            }
        }
        
        double test = max_x - min_x;
        if (test == 0.0) throw std::runtime_error("Scaling impossible: max col = min col");

        // Change data of our col
        for (size_t i = 0; i < n; i++) {
            x.at(j * n + i) = min + (x.at(j * n + i) - min_x) * (max - min) / (max_x - min_x);
        }
    }
    // Standard by default
    else {
        // Getting mean and data of col j
        double mean = 0.0;
        std::vector<double> col_j(n);
        for (size_t i = 0; i < n; i++) {
            col_j[i] = x.at(j * n + i);
            mean += x.at(j * n + i);
        }
        mean /= n;
        double sigma = std::sqrt(Stats::var(col_j));
        if (sigma == 0.0) throw std::runtime_error("Scaling impossible: sigma = 0");

        // Change data of our col
        for (size_t i = 0; i < n; i++) {
            x.at(j * n + i) = (x.at(j * n + i) - mean) / sigma;
        }
    }
}

void scaling(Dataframe& x, const std::string& col_name, const std::string& method, double min, double max) {

    // Find col
    auto headers = x.get_headers();
    auto idx = std::find(headers.begin(), headers.end(), col_name);

    if (idx != headers.end()) return scaling(x, static_cast<size_t>(idx - headers.begin()), method, min, max);
    else {
        throw std::invalid_argument(std::format("Column {} not found", col_name));
    }
}

void transform(Dataframe& x, size_t j, const std::string& method, double lambda) {

    if (j >= x.get_cols()) {
        throw std::invalid_argument("j >= nb_cols of x");
    }

    if (x.get_storage()) x.change_layout_inplace();

    size_t n = x.get_rows();
    if (method == "box_cox") {
        // Change data of our col
        for (size_t i = 0; i < n; i++) {
            x.at(j * n + i) = lambda == 0 ? std::log(x.at(j * n + i)) : (std::pow(x.at(j * n + i), lambda) - 1) / lambda;
        }
    }
    else if (method == "yeo_johnson") {
        // Change data of our col
        for (size_t i = 0; i < n; i++) {
            
            double val = x.at(j * n + i);
            if (val >= 0) {
                x.at(j * n + i) = lambda == 0 ? std::log(x.at(j * n + i) + 1) : (std::pow((x.at(j * n + i) + 1), lambda) - 1) / lambda;
            }
            else {
                x.at(j * n + i) = lambda == 2 ? -std::log(-x.at(j * n + i) + 1) : -(std::pow((-x.at(j * n + i) + 1), (2 - lambda)) - 1) / (2 - lambda);
            }
        }
    }
    else if (method == "power") {
        // Change data of our col
        for (size_t i = 0; i < n; i++) {
            x.at(j * n + i) = std::pow(x.at(j * n + i), lambda);
        }
    }
    // Log by default
    else {
        // Change data of our col
        for (size_t i = 0; i < n; i++) {
            x.at(j * n + i) = std::log(x.at(j * n + i));
        }
    }
}

void transform(Dataframe& x, const std::string& col_name, const std::string& method, double lambda) {

    // Find col
    auto headers = x.get_headers();
    auto idx = std::find(headers.begin(), headers.end(), col_name);

    if (idx != headers.end()) return transform(x, static_cast<size_t>(idx - headers.begin()), method, lambda);
    else {
        throw std::invalid_argument(std::format("Column {} not found", col_name));
    }
}
}