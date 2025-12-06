#include "Linalg/LinalgNaive.hpp"
#include "Linalg/Linalg.hpp"

namespace Linalg {
namespace Naive {

std::tuple<int, std::vector<double>, Dataframe> LU_decomposition(const Dataframe& df) {

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
        auto [max, idx] = std::tuple{-1, 0};
        for (size_t i = k; i < n; i++) {

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
            double mult = LU[k*n + i] / p;
            LU[k*n + i] = mult;

            // Update value in other cols 
            for (size_t j = k+1; j < n; j++) {
                LU[j*n + i] -= mult * LU[j*n + k];
            }
        }
    }
    
    return {nb_swaps, swaps, {n, n, false,  std::move(LU)}};
}

int triangular_matrix(const Dataframe& df) {

    size_t n = df.get_rows(); 
    bool is_trig_up = true;
    bool is_trig_down = true;

    // Triangular sup
    for (size_t i = 0; i < n && is_trig_up; i++) {
        for(size_t j = 0; j < i && is_trig_up; j++) {
            
            if (df.at(j*n + i) != 0) is_trig_up = false;
        }
    }

    // Triangular inf 
    for (size_t i = 0; i < n && is_trig_down; i++) {
        for(size_t j = i+1; j < n && is_trig_down; j++) {
            
            if (df.at(j*n + i) != 0) is_trig_down = false;
        }
    }

    if (is_trig_up && (is_trig_up && is_trig_down)) return 3; // Diag
    else if (is_trig_up) return 2; // Up
    else if (is_trig_down) return 1; // Down

    return 0; // Not triangular
}

std::tuple<double, std::vector<double>, Dataframe>determinant(Dataframe& df) {
    
    // Changing layout for better performances
    if (df.get_storage()){
        df.change_layout_inplace();
    }

    size_t rows = df.get_rows(), cols = df.get_cols();

    // First condition, Matrix(n,n)
    if (rows != cols) throw std::runtime_error("Need Matrix(n,n)");
    size_t n = rows;

    // Let's see if the matrix is diagonal or triangular 
    int test_v = triangular_matrix(df);
    if (test_v != 0) {
        
        double det = 1;
        for (size_t j = 0; j < n; j++) {
            
            det *= df.at(j*n + j); // Product of the diagonal
        }
        return {det, {static_cast<double>(test_v)}, {}};
    }
    else {

        auto [nb_swaps, swaps, LU] = LU_decomposition(df);

        double det = (nb_swaps % 2) ? -1.0 : 1.0;
        for (size_t j = 0; j < n; j++) {
            
            det *= LU.at(j*n + j); // Product of the diagonal
        }
        return {det, swaps, LU};
    }
}

Dataframe transpose(Dataframe& df) {

    size_t rows = df.get_cols(), cols = df.get_rows();
    size_t temp_row = df.get_rows(), temp_col = df.get_cols();

    std::vector<double> data;
    data.reserve(rows * cols);

    // Changing layout for better performances
    if (df.get_storage()){
        df.change_layout_inplace();
    }

    for (size_t i = 0; i < temp_row; i++) {
        for(size_t j = 0; j < temp_col; j++) {

            data.push_back(df.at(j*temp_row + i));
        }
    }   
    return {rows, cols, false, std::move(data), df.get_headers(), 
        df.get_encoder(), df.get_encodedCols()};
}

Dataframe sum(const Dataframe& df1, const Dataframe& df2, char op) {

    size_t m = df1.get_rows();
    size_t n = df1.get_cols();
    size_t o = df2.get_rows();
    size_t p = df2.get_cols();

    // Verify if we can sum them
    if (m != o || n != p) throw std::runtime_error("Need two Matrix of equal dimensions");

    // Condition to have better performances
    if (df1.get_storage() != df2.get_storage()) {
        throw std::runtime_error("Need two Matrix with the same storage Col-major or Row-major for performances purpose");
    }

    // New data
    std::vector<double> new_data(m * n);

    if (op == '+') {
        for (size_t i = 0; i < m*n; i++) {
            new_data[i] = df1.at(i) + df2.at(i);
        }
    }
    else if (op == '-') {
        for (size_t i = 0; i < m*n; i++) {
            new_data[i] = df1.at(i) - df2.at(i);
        }
    }

    return {m, n, false, std::move(new_data)};
}

Dataframe multiply(const Dataframe& df1, Dataframe& df2) {

    size_t m = df1.get_rows();
    size_t n = df1.get_cols();
    size_t o = df2.get_rows();
    size_t p = df2.get_cols();
    
    // Verify if we can multiply them
    if (n != o) throw std::runtime_error("Need df1 cols == df2 rows");

    // If row - row or col - col, we want to optimize
    if (df1.get_storage() == df2.get_storage()) {
        df2 = df2.change_layout();
    } 

    std::vector<double> data(m * p, 0.0);
    
    // row - col
    if (df1.get_storage()) {
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < p; j++) {

                double sum = 0.0;
                for (size_t k = 0; k < n; k++) {
                    // df1 row major
                    // df2 col major
                    sum += df1.at(i * n + k) * df2.at(j * o + k);
                }
                // Write it directly in col major
                data[j * m + i] = sum;
            }
        }
    }
    // col - row
    else {
        for (size_t i = 0; i < m; i++) {
            for (size_t k = 0; k < n; k++) {
                
                double val1 = df1.at(k*m + i); 
                for (size_t j = 0; j < p; j++) {
                    // df1 row major
                    // df2 col major
                    // Write it directly in col major
                    data[j*m + i] += val1 * df2.at(k*p + j); // Sequential read for performances
                }
            }
        }
    }
    
    // Return column - major
    return Dataframe(m, p, false, std::move(data), 
                     df1.get_headers(), df1.get_encoder(), df1.get_encodedCols());
}

Dataframe solveLU_inplace(const Dataframe& perm, Dataframe& LU) {

    size_t n = LU.get_cols();
    std::vector<double> y(n*n);

    // Store the Diag
    std::vector<double> diag_U(n);
    for (size_t i = 0; i < n; i++) {
        diag_U[i] = LU.at(i * n + i);
    }

    for (size_t k = 0; k < n; k++) {

        // Solving Ly = perm (No division because diag = 1)
        // Forward substitution
        for (size_t i = 0; i < n; i++) {

            double sum = 0.0;
            sum = perm.at(k*n + i);
            for (size_t j = 0; j < i; j++) {
                sum -= LU.at(j*n + i) * y[k*n + j];
            }
            y[k*n + i] = sum;
        }

        // Solving Ux = y 
        // Backward substitution
        for (int i = static_cast<int>(n)-1; i >= 0; i--) {
            for (size_t j = i+1; j < n; j++) {
                y[k*n + i] -= LU.at(j*n + i) * y[k*n + j];
            }
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
    Dataframe df_id = {n, n, false, std::move(id)};

    // If no LU matrix was returned, then the matrix is triangular
    if (LU.get_data().empty()) {
        std::vector<double> y(n*n, 0.0);

        // Diag
        if (swaps[0] == 3) {
            std::vector<double> y(n*n, 0.0);
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

                    y[k*n + i] = df_id.at(k*n + i);
                    for (size_t j = i+1; j < n; j++) {
                        y[k*n + i] -= df.at(j*n + i) * y[k*n + j];
                    }
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

        return solveLU_inplace(perm, LU);
    }
}

}
}