#include "Linalg/LinalgNaive.hpp"
#include "Linalg/Linalg.hpp"

namespace Linalg {
namespace Naive {

std::tuple<int, std::vector<double>, std::vector<double>> LU_decomposition(const Dataframe& df) {

    int nb_swaps = 0;
    size_t n = df.get_cols();
    std::vector<double> LU = df.get_data();

    // Permutation matrix is Id at first
    std::vector<double> swaps(n*n, 0.0);
    for (size_t i = 0; i < n; i++) {
        swaps[i*n + i] = 1;
    }

    for (size_t k = 0; k < n-1; k++) {

        // Partial pivot (get most important pivot and permutate lines)
        auto [max, idx] = std::tuple{LU[k*n + k], k};
        for (size_t i = k+1; i < n; i++) {

            double val = std::abs(LU[k*n + i]);
            if (max < val) {
                max =  val;
                idx = i;
            }
        }

        // Permutation of rows
        if (k != static_cast<size_t>(idx)) {
            for (size_t j = 0; j < n; j++) {
                std::swap(LU[j*n + k], LU[j*n + idx]);
                std::swap(swaps[j*n + k], swaps[j*n + idx]);
            }
            nb_swaps++;
        }

        // Pivot
        double p = LU[k*n + k];
        if (std::abs(p) < 1e-14) {
            // det = 0
            throw std::runtime_error("Singular Matrix <=> Det = 0");
        }

        // Then classical LU decomposition algorithm
        for (size_t i = k+1; i < n; i++) {

            // L value 
            double Lvalue = LU[k*n + i] / p;
            LU[k*n + i] = Lvalue;

            // Update value in other cols 
            for (size_t j = k+1; j < n; j++) {
                LU[j*n + i] -= Lvalue * LU[j*n + k];
            }
        }
    }
    return std::make_tuple(nb_swaps, std::move(swaps), std::move(LU));
}

Dataframe transpose(Dataframe& df) {

    size_t rows = df.get_cols(), cols = df.get_rows();
    size_t temp_row = df.get_rows(), temp_col = df.get_cols();

    // Changing layout for better performances later
    if (df.get_storage()){
        df.change_layout_inplace("Naive");
    }

    std::vector<double> data = Dataframe::transpose_naive(temp_row, temp_col, df.get_data());

    return {rows, cols, false, std::move(data), df.get_headers(), 
        df.get_encoder(), df.get_encodedCols()};
}

std::vector<double> sum(const std::vector<double>& v1, const std::vector<double>& v2, 
    size_t m, size_t n, char op = '+') {           

    // New data
    std::vector<double> new_data(m * n);

    if (op == '+') {
        for (size_t i = 0; i < m*n; i++) {
            new_data[i] = v1[i] + v2[i];
        }
    }
    else if (op == '-') {
        for (size_t i = 0; i < m*n; i++) {
            new_data[i] = v1[i] - v2[i];
        }
    }

    return new_data;
}

std::vector<double> multiply(const std::vector<double>& v1, const std::vector<double>& v2,
    size_t m, size_t n, size_t o, size_t p ) {
    
    // New data
    std::vector<double> new_data(m * p);
    
    // row - col
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {

            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                // df1 row major
                // df2 col major
                sum += v1[i * n + k] * v2[j * o + k];
            }
            // Write it directly in col major
            new_data[j * m + i] = sum;
        }
    }
    
    // Return column - major
    return new_data;
}

Dataframe solveLU_inplace(const std::vector<double>& perm, const std::vector<double>& LU, size_t n) {

    std::vector<double> y(n*n, 0.0);

    // Store the Diag
    std::vector<double> diag_U(n);
    for (size_t i = 0; i < n; i++) {
        diag_U[i] = LU[i * n + i];
    }

    for (size_t k = 0; k < n; k++) {

        // Solving Ly = perm (No division because diag = 1)
        // Forward substitution
        for (size_t i = 0; i < n; i++) {

            double sum = 0.0;
            sum = perm[k*n + i];
            for (size_t j = 0; j < i; j++) {
                sum -= LU[j*n + i] * y[k*n + j];
            }
            y[k*n + i] = sum;
        }

        // Solving Ux = y 
        // Backward substitution
        for (int i = static_cast<int>(n)-1; i >= 0; i--) {

            double sum = y[k*n + i];
            for (size_t j = i+1; j < n; j++) {
                sum -= LU[j*n + i] * y[k*n + j];
            }
            y[k*n + i] = sum;
            if (std::abs(y[k*n + i]) < 1e-14) y[k*n + i] = 0;
            else y[k*n + i] /= diag_U[i];
            
        }
    }
    return {n, n, false, std::move(y)};
}


Dataframe inverse(Dataframe& df) {

    auto [det, swaps, LU] = determinant(df);

    size_t n = df.get_cols();
    
    // Id matrix
    std::vector<double> id(n*n, 0.0);
    for (size_t i = 0; i < n; i++) {
        id[i*n + i] = 1;
    }
    Dataframe df_id = {n, n, true, std::move(id)};

    // If no LU matrix was returned, then the matrix is triangular
    if (LU.empty()) {
        std::vector<double> y(n*n, 0.0);

        // Diag
        if (swaps[0] == 3) {
            for (size_t i = 0; i < n; i++) {
               y[i*n + i] = 1 / df.at(i*n+i);
            }
            return {n, n, false, std::move(y)};
        }
        // Up
        else if (swaps[0] == 2) {

            // Solving Uy = Id 
            // Backward substitution
            for (size_t k = 0; k < n; k++) {
                for (int i = static_cast<int>(n)-1; i >= 0; i--) {

                    double sum = df_id.at(k*n + i);
                    for (size_t j = i+1; j < n; j++) {
                        sum -= df.at(j*n + i) * y[k*n + j];
                    }
                    y[k*n + i] = sum;
                    if (std::abs(y[k*n + i]) < 1e-14) y[k*n + i] = 0;
                    else y[k*n + i] /= df.at(i*n+i);
                }
            }
            return {n, n, false, std::move(y)};
        }
        // Down
        else {

            // Solving Ly = Id
            // Forward substitution
            for (size_t k = 0; k < n; k++) {
                for (size_t i = 0; i < n; i++) {

                    double sum = 0.0;
                    sum = df_id.at(k*n + i);
                    for (size_t j = 0; j < i; j++) {
                        sum -= df.at(j*n + i) * y[k*n + j];
                    }
                    y[k*n + i] = sum / df.at(i*n + i);
                }
            }
            return {n, n, false, std::move(y)};
        }
    }
    else {
        
        // Permutation matrix
        Dataframe df_swaps = {n, n, false, std::move(swaps)};
        Dataframe perm = multiply(df_swaps, df_id); 

        return solveLU_inplace(perm.get_data(), LU, n);
    }
}

}
}