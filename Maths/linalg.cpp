#include "Linalg.hpp"


Dataframe Linalg::transpose_naive(const Dataframe& df) {

    size_t rows = df.get_cols(), cols = df.get_rows();
    size_t temp_row = df.get_rows(), temp_col = df.get_cols();

    std::vector<double> data;
    data.reserve(rows * cols);

    if (df.get_storage()){
        for (size_t i = 0; i < temp_col; i++) {
            for(size_t j = 0; j < temp_row; j++) {

                data.push_back(df.at(j*temp_col + i));

            }
        }
    }
    else {
        for (size_t i = 0; i < temp_row; i++) {
            for(size_t j = 0; j < temp_col; j++) {

                data.push_back(df.at(j*temp_row + i));

            }
        }   
    }
    return {rows, cols, df.get_storage(), std::move(data), df.get_headers(), 
        df.get_encoder(), df.get_encodedCols()};
}


Dataframe Linalg::multiply_naive(const Dataframe& df1, Dataframe& df2) {

    size_t m = df1.get_rows();
    size_t n = df1.get_cols();
    size_t o = df2.get_rows();
    size_t p = df2.get_cols();
    
    // Verify if we can multiply them
    assert(n == o);

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
