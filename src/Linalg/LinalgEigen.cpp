#include "Linalg/LinalgEigen.hpp"
#include "Linalg/Linalg.hpp"

namespace Linalg {
namespace EigenNP {

Dataframe transpose(Dataframe& df) {

    size_t rows = df.get_cols(), cols = df.get_rows();
    size_t temp_row = df.get_rows(), temp_col = df.get_cols();

    // Changing layout for better performances later
    if (df.get_storage()){
        df.change_layout_inplace("Eigen");
    }

    std::vector<double> data = Dataframe::transpose_eigen(temp_row, temp_col, df.get_data());

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
    if ((df1.get_storage() != df2.get_storage()) && df1.get_storage()) {
        throw std::runtime_error("Need two Matrix with the same storage and Col-major for performances purpose");
    }

    // New data
    std::vector<double> new_data(m * n);
    Eigen::Map<Eigen::MatrixXd> res(new_data.data(), m, n);

    if (op == '+') {
        res = df1.asEigen() + df2.asEigen();
    }
    else if (op == '-') {
        res = df1.asEigen() - df2.asEigen();
    }

    return {m, n, false, std::move(new_data)};
}

Dataframe multiply(const Dataframe& df1, const Dataframe& df2) {

    size_t m = df1.get_rows();
    size_t n = df1.get_cols();
    size_t o = df2.get_rows();
    size_t p = df2.get_cols();
    
    // Verify if we can multiply them
    if (n != o) throw std::runtime_error("Need df1 cols == df2 rows");

    // Condition to have better performances
    if ((df1.get_storage() != df2.get_storage()) && df1.get_storage()) {
        throw std::runtime_error("Need two Matrix with the same storage and Col-major for performances purpose");
    }

    std::vector<double> new_data(m * p);
    Eigen::Map<Eigen::MatrixXd> res(new_data.data(), m, p);

    res = df1.asEigen() * df2.asEigen();

    // Return column - major
    return Dataframe(m, p, false, std::move(new_data), 
                     df1.get_headers(), df1.get_encoder(), df1.get_encodedCols());
}

Dataframe inverse(Dataframe& df) {
    size_t n = df.get_cols();

    std::vector<double> new_data(n * n);
    Eigen::Map<Eigen::MatrixXd> res(new_data.data(), n, n);

    res = df.asEigen().inverse();

    return Dataframe(n, n, false, std::move(new_data));
}

}
}