#include "TestSuite.hpp"
#include "Linalg/Linalg.hpp"

using namespace std;
using namespace Linalg;

#ifdef __AVX2__

// Testing Sum AVX2_threaded
void sum_avx2_th(Dataframe& df1, Dataframe& df2, char op, const std::vector<double>& res) {

    Dataframe df = Operations::sum(df1, df2, op);

    ASSERT_EQ(df.get_data(), res)
}

// Testing AVX2_threaded Multiply
void multiply_avx2_th(Dataframe& df, Dataframe& df_t, const vector<double>& res) {
    
    Dataframe df_bis_t = df_t.change_layout("AVX2_threaded"); 
    
    // Row - col
    Dataframe df_mult = Operations::multiply(df_bis_t, df);

    ASSERT_EQ_VEC_EPS(df_mult.get_data(), res)
}

// Testing AVX2_threaded Inverse LU
void inverse_avx2_th(Dataframe& df, Dataframe& df_t, const std::vector<double>& res) {
    
    Dataframe df_bis_t = df_t.change_layout("AVX2_threaded"); 

    // Row - col
    Dataframe df_mult = Operations::multiply(df_bis_t, df);

    // Inv
    Dataframe df_inv = Operations::inverse(df_mult);

    ASSERT_EQ_VEC_EPS(df_inv.get_data(), res)
}


void tests_avx2_th() {

    Operations::set_backend(Backend::AVX2_THREADED);

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
    TestSuite::Tests tests_avx2_th;

    tests_avx2_th.add_test(
        bind(sum_avx2_th, iris, iris, '+', iris_sum.get_data()), 
        "Testing AVX2_threaded Sum v1"
    );

    tests_avx2_th.add_test(
        bind(sum_avx2_th, mat, mat, '+', mat_sum.get_data()), 
        "Testing AVX2_threaded Sum v2"
    );

    tests_avx2_th.add_test(
        bind(multiply_avx2_th, iris, iris_t, iris_mult.get_data()), 
        "Testing AVX2_threaded Multiply v1"
    );

    tests_avx2_th.add_test(
        bind(multiply_avx2_th, mat, mat_t, mat_mult.get_data()), 
        "Testing AVX2_threaded Multiply v2"
    );

    tests_avx2_th.add_test(
        bind(inverse_avx2_th, iris, iris_t, iris_inv.get_data()), 
        "Testing AVX2_threaded Inverse v1"
    );

    tests_avx2_th.add_test(
        bind(inverse_avx2_th, mat, mat_t, mat_inv.get_data()), 
        "Testing AVX2_threaded Inverse v2"
    );

    cout << "Testing AVX2_threaded functions:" << endl;
    tests_avx2_th.run_all();
}
#endif