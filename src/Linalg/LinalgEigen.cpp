#include "Linalg/LinalgEigen.hpp"

namespace Linalg::EigenNP {

std::vector<double> transpose(const std::vector<double>& v1,  
    size_t v1_rows, size_t v1_cols) {

    std::vector<double> new_data = Dataframe::transpose_eigen(v1_rows, v1_cols, v1);

    return new_data;
}

std::vector<double> sum(const std::vector<double>& v1, const std::vector<double>& v2, 
    size_t m, size_t n, char op = '+') { 

    // New data
    std::vector<double> new_data(m * n);
    Eigen::Map<Eigen::MatrixXd> res(new_data.data(), m, n);
    Eigen::Map<const Eigen::MatrixXd> mat1(v1.data(), m, n);
    Eigen::Map<const Eigen::MatrixXd> mat2(v2.data(), m, n);

    if (op == '+') {
        res = mat1 + mat2;
    }
    else if (op == '-') {
        res = mat1 - mat2;
    }

    return new_data;
}

std::vector<double> multiply(const std::vector<double>& v1, const std::vector<double>& v2,
    size_t m, size_t n, size_t o, size_t p ) {

    std::vector<double> new_data(m * p);
    Eigen::Map<Eigen::MatrixXd> res(new_data.data(), m, p);
    Eigen::Map<const Eigen::MatrixXd> mat1(v1.data(), m, n);
    Eigen::Map<const Eigen::MatrixXd> mat2(v2.data(), o, p);

    res = mat1 * mat2;

    // Return column - major
    return new_data;
}

Dataframe inverse(Dataframe& df) {
    size_t n = df.get_cols();

    std::vector<double> new_data(n * n);
    Eigen::Map<Eigen::MatrixXd> res(new_data.data(), n, n);

    res = df.asEigen().inverse();

    return Dataframe(n, n, false, std::move(new_data));
}
}