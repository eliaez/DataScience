#include "Linalg.hpp"


Dataframe Linalg::transpose_naive(const Dataframe& df) {

    size_t rows, cols;

    if (df.get_storage()) {
        rows = df.get_cols(); 
        cols = df.get_rows();
    }
    else {
        rows = df.get_rows(); 
        cols = df.get_cols();
    }

    std::vector<double> data;
    data.reserve(rows * cols);

    for (size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {

            if (df.get_storage()) data.push_back(df(j,i));
            else data.push_back(df(i,j));
        }
    }

    // Keeping original dimensions despite transpose for practical use 
    // Operator & others functions are implemented in consequence 
    return {df.get_rows(), df.get_cols(), !df.get_storage(), std::move(data), df.get_headers(), 
        df.get_encoder(), df.get_encodedCols()};
}

const Dataframe& Linalg::multiply_config(const Dataframe& df1, 
    const Dataframe& df2, Dataframe& temp_storage) {
    
    // Either way both are block matrix or row-col / col-row in the near future
    assert(df1.get_cols() == df2.get_cols());

    bool df1_config = df1.get_storage();
    bool df2_config = df2.get_storage();
    
    // row - row or col - col
    if (df1_config == df2_config) {

        temp_storage = transpose_naive(df2); // Changed to row - col or col - row
        return temp_storage;    
    }

    return df2;
}

Dataframe Linalg::multiply_naive(const Dataframe& df1, const Dataframe& df2) {
    
    size_t m = df1.get_rows();
    size_t k = df1.get_cols();
    size_t n = df2.get_cols();
    
    std::vector<double> data(m * n, 0.0);
    
    Dataframe temp_storage;
    const Dataframe& temp = multiply_config(df1, df2, temp_storage);
    
    if (df1.get_storage()) {
        // row - col 
        for (size_t j = 0; j < n; j++) {
            for (size_t i = 0; i < m; i++) {
                for (size_t p = 0; p < k; p++) {
                    data[j * m + i] += df1(i, p) * temp(j, p);
                }
            }
        }
    }
    else {
        // col - row 
        for (size_t j = 0; j < n; j++) {
            for (size_t i = 0; i < m; i++) {
                for (size_t p = 0; p < k; p++) {
                    data[j * m + i] += df1(i, p) * temp(p, j);
                }
            }
        }
    }
    
    // Toujours retourner col-major (false)
    return Dataframe(m, n, false, std::move(data), 
                     df1.get_headers(), df1.get_encoder(), df1.get_encodedCols());
}
