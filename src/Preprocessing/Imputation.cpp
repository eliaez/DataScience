#include <cmath>
#include <stdexcept>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Preprocessing/Imputation.hpp"

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

        double most = Utils::mostFrequent(col_j);

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
}