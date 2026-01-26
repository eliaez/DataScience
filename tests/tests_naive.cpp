#include "TestSuite.hpp"
#include "Linalg/Linalg.hpp"
#include "Data/Data.hpp"

using namespace std;
using namespace Linalg;

// Testing Sum
void sum_naive(Dataframe& df1, Dataframe& df2, char op, const vector<double>& res) {

    Dataframe df = Operations::sum(df1, df2, op);

    ASSERT_EQ(df.get_data(), res)
}

// Testing Naive Multiply v1
void multiply_naive_v1(Dataframe& df1, Dataframe& df2, const vector<double>& res) {
    
    Dataframe df1_temp = df1.change_layout("Naive"); 
    
    // Row - col
    Dataframe df = Operations::multiply(df1_temp, df2);

    ASSERT_EQ(df.get_data(), res)
}

// Testing Naive Multiply v2
void multiply_naive_v2(Dataframe& mat, Dataframe& mat_t, const std::vector<double>& mat_mult) {
    
    Dataframe mat_bis_t = mat_t.change_layout("Naive"); 
    
    // Row - col
    Dataframe df = Operations::multiply(mat_bis_t, mat);

    ASSERT_EQ_VEC_EPS(df.get_data(), mat_mult)
}

// Testing Naive Inverse - Error det = 0
void inverse_naive_v1(Dataframe& df1) {
    Dataframe df = Operations::inverse(df1);
}

// Testing Naive Inverse Triangular
void inverse_naive_v2(Dataframe& df1, const vector<double>& res) {
    
    Dataframe df = Operations::inverse(df1);

    ASSERT_EQ(df.get_data(), res)
}

// Testing Naive Inverse LU
void inverse_naive_v3(Dataframe& mat, Dataframe& mat_t, const std::vector<double>& mat_inv) {
    
    Dataframe mat_bis_t = mat_t.change_layout("Naive"); 

    // Row - col
    Dataframe df = Operations::multiply(mat_bis_t, mat);

    // Inv
    Dataframe df_inv = Operations::inverse(df);

    ASSERT_EQ_VEC_EPS(df_inv.get_data(), mat_inv)
}


void tests_naive() {

    Operations::set_backend(Backend::NAIVE);

    // Initialization of our data 
    size_t n = 4;

    vector<double> v1 = {1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16};
    vector<double> v2 = {2,4,1,0,0,5,3,2,1,0,6,4,3,2,0,1};
    vector<double> v3 = {1,0,0,0,2,1,0,0,3,2,1,0,4,3,2,1};

    Dataframe df1 = {n, n, false, v1};
    Dataframe df2 = {n, n, false, v2};
    Dataframe df3 = {n, n, false, v3};

    vector<double> sum = {-1,1,8,13,2,1,7,12,2,7,5,11,1,6,12,15};
    vector<double> mult = {13,41,69,97,27,67,107,147,35,79,123,167,11,35,59,83};
    vector<double> inv3 = {1,0,0,0,-2,1,0,0,1,-2,1,0,0,1,-2,1};

    Dataframe mat = CsvHandler::loadCsv("../tests/datasets/mat.csv", ',', false);
    Dataframe mat_t = CsvHandler::loadCsv("../tests/datasets/mat_t.csv", ',', false);
    Dataframe mat_mult = CsvHandler::loadCsv("../tests/datasets/mat_mult.csv", ',', false);
    Dataframe mat_inv = CsvHandler::loadCsv("../tests/datasets/mat_multinv.csv", ',', false);

    // Add tests
    TestSuite::Tests tests_naive;

    tests_naive.add_test(
        bind(sum_naive, df1, df2, '-', sum), 
        "Testing Naive Sum"
    );

    tests_naive.add_test(
        bind(multiply_naive_v1, df1, df2, mult), 
        "Testing Naive Multiply v1"
    );

    tests_naive.add_test(
        bind(multiply_naive_v2, mat, mat_t, mat_mult.get_data()), 
        "Testing Naive Multiply v2"
    );

    tests_naive.add_test(
        bind(inverse_naive_v1, df1), 
        "Testing Naive Inverse v1 - Error"
    );

    tests_naive.add_test(
        bind(inverse_naive_v2, df3, inv3), 
        "Testing Naive Inverse v2 - Triangular"
    );

    tests_naive.add_test(
        bind(inverse_naive_v3, mat, mat_t, mat_inv.get_data()), 
        "Testing Naive Inverse v3 - LU"
    );

    cout << "Testing Naive functions:" << endl;
    tests_naive.run_all();
}