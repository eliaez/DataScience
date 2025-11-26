#include "Linalg.hpp"


Dataframe Linalg::transpose_naive(const Dataframe& df) {

    size_t rows = df.get_rows(), cols = df.get_cols();
    std::vector<double> data;
    data.reserve(rows * cols);

    for (size_t i = 0; i < cols; i++) {
        for(size_t j = 0; j < rows; j++) {

            data.push_back(df(j,i));
        }
    }

    return {rows, cols, !df.get_storage(), std::move(data), df.get_headers(), 
        df.get_encoder(), df.get_encodedCols()};
}


Dataframe Linalg::multiply_naive(const Dataframe& df1, const Dataframe& df2) {
    
}