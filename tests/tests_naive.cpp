#include "Utils/TestSuite.hpp"
#include "Linalg/Linalg.hpp"

using namespace std;
using namespace Linalg;

// Testing Naive Transpose col-major
void transpose_t(Dataframe& iris, const vector<double>& res) {

    Dataframe iris_t = Operations::transpose(iris);

    ASSERT_EQ(iris_t.get_data(), res)
}

// Testing Sum
void sum_t(Dataframe& df1, Dataframe& df2, char op, const vector<double>& res) {

    Dataframe df = Operations::sum(df1, df2, op);

    ASSERT_EQ(df.get_data(), res)
}

// Testing Naive Multiply
void multiply_t(Dataframe& df1, Dataframe& df2, const vector<double>& res) {
    
    Dataframe df1_temp = df1.change_layout(); 
    Dataframe df = Operations::multiply(df1_temp, df2);

    ASSERT_EQ(df.get_data(), res)
}

// Testing Naive Inverse - Error det = 0
void inverse_v1(Dataframe& df1) {
    Dataframe df = Operations::inverse(df1);
}

// Testing Naive Inverse Triangular
void inverse_v2(Dataframe& df1, const vector<double>& res) {
    
    Dataframe df = Operations::inverse(df1);

    ASSERT_EQ(df.get_data(), res)
}

// Testing Naive Inverse LU
void inverse_v3(Dataframe& df1, const vector<double>& res) {
    
    Dataframe df = Operations::inverse(df1);

    ASSERT_EQ(df.get_data(), res)
}


void tests_naive() {

    Operations::set_backend(Backend::NAIVE);

    // Initialization of our data 
    size_t m = 3, n = 4;

    vector<double> iris_colmajor = {5.1,4.9,4.7,3.5,3.0,3.2,1.4,1.4,1.3,0.2,0.2,0.2};
    vector<double> iris_t = {5.1,3.5,1.4,0.2,4.9,3.0,1.4,0.2,4.7,3.2,1.3,0.2};

    Dataframe iris = {m, n, false, iris_colmajor};

    vector<double> v1 = {1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16};
    vector<double> v2 = {2,4,1,0,0,5,3,2,1,0,6,4,3,2,0,1};
    vector<double> v3 = {1,0,0,0,2,1,0,0,3,2,1,0,4,3,2,1};
    vector<double> v4 = {3,2,1,0,2,2,1,0,1,1,1,0,0,0,0,1};

    vector<double> sum = {-1,1,8,13,2,1,7,12,2,7,5,11,1,6,12,15};
    vector<double> mult = {13,41,69,97,27,67,107,147,35,79,123,167,11,35,59,83};
    vector<double> inv3 = {1,0,0,0,-2,1,0,0,1,-2,1,0,0,1,-2,1};
    vector<double> inv4 = {1,-0.99999999999999989,0,0,-1,2,-1,0,0,-1.0000000000000002,2,0,0,0,0,1};

    Dataframe df1 = {n, n, false, v1};
    Dataframe df2 = {n, n, false, v2};
    Dataframe df3 = {n, n, false, v3};
    Dataframe df4 = {n, n, false, v4};

    // Add tests
    TestSuite::Tests tests_naive;

    tests_naive.add_test(
        bind(transpose_t, iris, iris_t), 
        "Naive Transpose col-major"
    );

    tests_naive.add_test(
        bind(sum_t, df1, df2, '-', sum), 
        "Testing Naive Sum"
    );

    tests_naive.add_test(
        bind(multiply_t, df1, df2, mult), 
        "Testing Naive Multiply"
    );

    tests_naive.add_test(
        bind(inverse_v1, df1), 
        "Testing Naive Inverse v1 - Error"
    );

    tests_naive.add_test(
        bind(inverse_v2, df3, inv3), 
        "Testing Naive Inverse v2 - Triangular"
    );

    tests_naive.add_test(
        bind(inverse_v3, df4, inv4), 
        "Testing Naive Inverse v3 - LU"
    );

    cout << "Testing Naive functions:" << endl;
    tests_naive.run_all();
}