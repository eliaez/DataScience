#include "Linalg.hpp"

std::tuple<int, Dataframe> Linalg::LU_decomposition(const Dataframe& df) {

    int nb_swaps = 0;
    size_t n = df.get_cols();
    std::vector<double> LU = df.get_data();

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
        if (k != idx) {
            for (size_t j = 0; j < n; j++) {
                std::swap(LU[j*n + k], LU[j*n + idx]);
            }
            nb_swaps++;
        }

        // Pivot
        double p = LU[k*n + k];
        if (std::abs(p) < 1e-14) {
            // det = 0
            throw std::runtime_error("Singular Matrix");
        }

        // Then classical LU decomposition algorithm
        for (size_t i = k+1; i < n; i++) {

            // L value 
            double mult = LU[k*n + i] / p;
            LU[k*n + i] = mult;

            // Update value in other cols 
            for (size_t j = k; j < n; j++) {
                LU[j*n + i] -= mult * LU[j*n + k];
            }
        }
    }
    
    return {nb_swaps, {n, n, false,  std::move(LU)}};
}

bool Linalg::triangular_matrix(const Dataframe& df) {

    size_t n = df.get_rows(); 

    // Triangular sup
    for (size_t j = 0; j < n; j++) {
        for(size_t i = 0; i < j; i++) {
            
            if (df.at(j*n + i) != 0) return false;
        }
    }

    // Triangular inf 
    for (size_t j = 0; j < n; j++) {
        for(size_t i = j+1; i < n; i++) {
            
            if (df.at(j*n + i) != 0) return false;
        }
    }
    return true;
}

std::tuple<double, Dataframe> Linalg::determinant_naive(Dataframe& df) {
    
    if (df.get_storage()){
        df.change_layout_inplace();
    }

    size_t rows = df.get_rows(), cols = df.get_cols();

    // First condition, block matrix
    assert(rows == cols);
    size_t n = rows;

    if (n == 2) {
        // a*d - b*c
        double det = df.at(0)*df.at(3)-df.at(2)*df.at(1);
        return {det, {}};
    }
    else if (n == 3) {
        // a*(e*i - h*f) + b*(f*g - d*i) + c*(d*h - g*e) 
        double det = (df.at(0)*(df.at(4)*df.at(8) - df.at(5)*df.at(7)) 
                    + df.at(3)*(df.at(7)*df.at(2) - df.at(1)*df.at(8))
                    + df.at(6)*(df.at(1)*df.at(5) - df.at(2)*df.at(4)));
        return {det, {}};
    }
    else {
        // Let's see if the matrix is diagonal or triangular 
        if (triangular_matrix(df)) {
            
            double det = 1;
            for (size_t j = 0; j < n; j++) {
                
                det *= df.at(j*n + j); // Product of the diagonal
            }
            return {det, {}};
        }
        else {

            auto [nb_swaps, LU] = LU_decomposition(df);

            double det = pow(-1, nb_swaps);
            for (size_t j = 0; j < n; j++) {
                
                det *= LU.at(j*n + j); // Product of the diagonal
            }
            return {det, LU};
        }
    }
}

Dataframe Linalg::transpose_naive(Dataframe& df) {

    size_t rows = df.get_cols(), cols = df.get_rows();
    size_t temp_row = df.get_rows(), temp_col = df.get_cols();

    std::vector<double> data;
    data.reserve(rows * cols);

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


Dataframe Linalg::multiply_naive(const Dataframe& df1, Dataframe& df2) {

    size_t m = df1.get_rows();
    size_t n = df1.get_cols();
    size_t o = df2.get_rows();
    size_t p = df2.get_cols();
    
    // Verify if we can multiply them
    assert(n == o);

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
