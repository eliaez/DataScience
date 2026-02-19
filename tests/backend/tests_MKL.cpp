#include "TestSuite.hpp"
#include "Linalg/Linalg.hpp"
#include "Data/Data.hpp"

using namespace std;
using namespace Linalg;

#ifdef USE_MKL

// Testing Sum MKL
void sum_mkl(Dataframe& df1, Dataframe& df2, char op, const std::vector<double>& res) {

    Dataframe df = Operations::sum(df1, df2, op);

    ASSERT_EQ(df.get_data(), res)
}

// Testing MKL Multiply
void multiply_mkl(Dataframe& df, Dataframe& df_t, const vector<double>& res) {
    
    // Col - col
    Dataframe df_mult = Operations::multiply(df_t, df);

    ASSERT_EQ_VEC_EPS(df_mult.get_data(), res, 1e-9)
}

// Testing MKL Inverse LU
void inverse_mkl(Dataframe& df, Dataframe& df_t, const std::vector<double>& res) {
    
    // Col - col
    Dataframe df_mult = Operations::multiply(df_t, df);

    // Inv
    Dataframe df_inv = Operations::inverse(df_mult);

    ASSERT_EQ_VEC_EPS(df_inv.get_data(), res, 1e-9)
}


void tests_mkl() {

    Operations::set_backend(Backend::MKL);

    Dataframe iris = CsvHandler::loadCsv("../tests/datasets/backend/iris.csv");
    Dataframe iris_t = CsvHandler::loadCsv("../tests/datasets/backend/iris_t.csv", ',', false);
    Dataframe iris_sum = CsvHandler::loadCsv("../tests/datasets/backend/iris_sum.csv", ',', false);
    Dataframe iris_mult = CsvHandler::loadCsv("../tests/datasets/backend/iris_mult.csv", ',', false);
    Dataframe iris_inv = CsvHandler::loadCsv("../tests/datasets/backend/iris_multinv.csv", ',', false);

    Dataframe mat = CsvHandler::loadCsv("../tests/datasets/backend/mat.csv", ',', false);
    Dataframe mat_t = CsvHandler::loadCsv("../tests/datasets/backend/mat_t.csv", ',', false);
    Dataframe mat_sum = CsvHandler::loadCsv("../tests/datasets/backend/mat_sum.csv", ',', false);
    Dataframe mat_mult = CsvHandler::loadCsv("../tests/datasets/backend/mat_mult.csv", ',', false);
    Dataframe mat_inv = CsvHandler::loadCsv("../tests/datasets/backend/mat_multinv.csv", ',', false);

    // Add tests
    TestSuite::Tests tests_mkl;

    tests_mkl.add_test(
        bind(sum_mkl, iris, iris, '+', iris_sum.get_data()), 
        "Testing MKL Sum v1"
    );

    tests_mkl.add_test(
        bind(sum_mkl, mat, mat, '+', mat_sum.get_data()), 
        "Testing MKL Sum v2"
    );

    tests_mkl.add_test(
        bind(multiply_mkl, iris, iris_t, iris_mult.get_data()), 
        "Testing MKL Multiply v1"
    );

    tests_mkl.add_test(
        bind(multiply_mkl, mat, mat_t, mat_mult.get_data()), 
        "Testing MKL Multiply v2"
    );

    tests_mkl.add_test(
        bind(inverse_mkl, iris, iris_t, iris_inv.get_data()), 
        "Testing MKL Inverse v1"
    );

    tests_mkl.add_test(
        bind(inverse_mkl, mat, mat_t, mat_inv.get_data()), 
        "Testing MKL Inverse v2"
    );

    cout << "Testing MKL functions:" << endl;
    tests_mkl.run_all();
}
#endif