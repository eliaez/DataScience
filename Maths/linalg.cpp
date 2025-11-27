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


Dataframe Linalg::multiply_naive(const Dataframe& df1, const Dataframe& df2) {

    size_t m = df1.get_rows();
    size_t n = df1.get_cols();
    size_t o = df2.get_rows();
    size_t p = df2.get_cols();
    
    if (m != o) {

        // Verify if we can multiply them
        assert(n == o);
    }

    std::vector<double> data(m * p, 0.0);
    
    for (size_t i = 0; i < m; i++) {

        for (size_t j = 0; j < p; j++) {

            for (size_t k = 0; k < o; k++) {

                data[i * p + j] += df1(j, k) * df2(k,i);
            }
        }
    }
    
    // Return column - major
    return Dataframe(m, p, false, std::move(data), 
                     df1.get_headers(), df1.get_encoder(), df1.get_encodedCols());
}
