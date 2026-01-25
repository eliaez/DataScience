#include "TestSuite.hpp"
#include "Linalg/Linalg.hpp"

using namespace std;
using namespace Linalg;

// Testing Sum Eigen
void sum_eigen(Dataframe& df1, Dataframe& df2, char op, const std::vector<double>& res) {

    Dataframe df = Operations::sum(df1, df2, op);

    ASSERT_EQ(df.get_data(), res)
}

// Testing Eigen Multiply
void multiply_eigen(Dataframe& df, Dataframe& df_t, const vector<double>& res) {
    
    // Col - col
    Dataframe df_mult = Operations::multiply(df_t, df);

    ASSERT_EQ_VEC_EPS(df_mult.get_data(), res)
}

// Testing Eigen Inverse LU
void inverse_eigen(Dataframe& df, Dataframe& df_t, const std::vector<double>& res) {
    
    // Col - col
    Dataframe df_mult = Operations::multiply(df_t, df);

    // Inv
    Dataframe df_inv = Operations::inverse(df_mult);

    ASSERT_EQ_VEC_EPS(df_inv.get_data(), res)
}


void tests_eigen() {

    Operations::set_backend(Backend::EIGEN);

    Dataframe iris = CsvHandler::loadCsv("../tests/datasets/iris.csv");
    Dataframe iris_t = CsvHandler::loadCsv("../tests/datasets/iris_t.csv", ',', false);
    Dataframe iris_sum = CsvHandler::loadCsv("../tests/datasets/iris_sum.csv", ',', false);
    Dataframe iris_mult = CsvHandler::loadCsv("../tests/datasets/iris_mult.csv", ',', false);
    Dataframe iris_inv = CsvHandler::loadCsv("../tests/datasets/iris_multinv.csv", ',', false);

    Dataframe mat = CsvHandler::loadCsv("../tests/datasets/mat.csv", ',', false);
    Dataframe mat_t = CsvHandler::loadCsv("../tests/datasets/mat_t.csv", ',', false);
    Dataframe mat_sum = CsvHandler::loadCsv("../tests/datasets/mat_sum.csv", ',', false);
    Dataframe mat_mult = CsvHandler::loadCsv("../tests/datasets/mat_mult.csv", ',', false);
    Dataframe mat_inv = CsvHandler::loadCsv("../tests/datasets/mat_multinv.csv", ',', false);

    // Add tests
    TestSuite::Tests tests_eigen;

    tests_eigen.add_test(
        bind(sum_eigen, iris, iris, '+', iris_sum.get_data()), 
        "Testing Eigen Sum v1"
    );

    tests_eigen.add_test(
        bind(sum_eigen, mat, mat, '+', mat_sum.get_data()), 
        "Testing Eigen Sum v2"
    );

    tests_eigen.add_test(
        bind(multiply_eigen, iris, iris_t, iris_mult.get_data()), 
        "Testing Eigen Multiply v1"
    );

    tests_eigen.add_test(
        bind(multiply_eigen, mat, mat_t, mat_mult.get_data()), 
        "Testing Eigen Multiply v2"
    );

    tests_eigen.add_test(
        bind(inverse_eigen, iris, iris_t, iris_inv.get_data()), 
        "Testing Eigen Inverse v1"
    );

    tests_eigen.add_test(
        bind(inverse_eigen, mat, mat_t, mat_inv.get_data()), 
        "Testing Eigen Inverse v2"
    );

    cout << "Testing Eigen functions:" << endl;
    tests_eigen.run_all();
}