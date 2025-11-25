#include "linalg.hpp"


Dataframe Linalg::transpose_naive(const Dataframe& df) {

    // Class Dataframe variables
    size_t rows = df.get_rows(), cols = df.get_cols();
    std::vector<double> data;
    data.reserve(rows * cols);

    for (size_t i = 0; i < cols; i++) {
        for(size_t j = 0; j < rows; j++) {

            data.push_back(df.at_row_major(j,i));
        }
    }

    return {rows, cols, std::move(data), df.get_headers(), df.get_encoder(), df.get_encodedCols()};
}
