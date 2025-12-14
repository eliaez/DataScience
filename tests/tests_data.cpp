#include "Utils/TestSuite.hpp"
#include "Data/Data.hpp"

using namespace std;

// Testing data after change_layout Naive
void changelayout_naive(Dataframe iris, const Dataframe& iris_t) {
    
    Dataframe iris_bis_t = iris.change_layout("Naive");

    ASSERT_EQ(iris_bis_t.get_data(), iris_t.get_data())
}

// Testing data after change_layout AVX2
void changelayout_avx2_v1(Dataframe iris, const Dataframe& iris_t) {
    
    Dataframe iris_bis_t = iris.change_layout("AVX2");

    ASSERT_EQ(iris_bis_t.get_data(), iris_t.get_data())
}

// Testing data after change_layout AVX2 (different data size)
void changelayout_avx2_v2(Dataframe mat, const Dataframe& mat_t) {
    
    Dataframe mat_bis_t = mat.change_layout("AVX2");

    ASSERT_EQ(mat_bis_t.get_data(), mat_t.get_data())
}

// Testing the transfer of a column from a Df to another new one
void transfercol_t(Dataframe& iris, const string& col,
    const Dataframe& iris_x, const Dataframe& iris_y) {

    Dataframe y = iris.transfer_col(col);
    
    // Data new df
    ASSERT_EQ(y.get_data(), iris_y.get_data())

    // Headers new df
    ASSERT_EQ(y.get_headers()[0], col)

    // Data former df
    ASSERT_EQ(iris.get_data(), iris_x.get_data())

    // Headers former df
    ASSERT_EQ(iris.get_headers(), iris_x.get_headers())
}

void tests_data() {

    // Initialization of our data 
    Dataframe iris = CsvHandler::loadCsv("../tests/datasets/iris.csv");
    Dataframe iris_t = CsvHandler::loadCsv("../tests/datasets/iris_t.csv");
    Dataframe iris_x = CsvHandler::loadCsv("../tests/datasets/iris_x.csv");
    Dataframe iris_y = CsvHandler::loadCsv("../tests/datasets/iris_y.csv");

    Dataframe mat = CsvHandler::loadCsv("../tests/datasets/mat.csv", ',', false);
    Dataframe mat_t = CsvHandler::loadCsv("../tests/datasets/mat_t.csv", ',', false);

    // Add tests
    TestSuite::Tests tests_data;

    tests_data.add_test(
        bind(changelayout_naive, iris, iris_t), 
        "Change layout Naive"
    );

    tests_data.add_test(
        bind(changelayout_avx2_v1, iris, iris_t), 
        "Change layout AVX2 v1"
    );

    tests_data.add_test(
        bind(changelayout_avx2_v2, mat, mat_t), 
        "Change layout AVX2 v2"
    );
   
    tests_data.add_test(
        bind(transfercol_t, iris, "target", iris_x, iris_y), 
        "Transfer column"
    );

    cout << "Testing Dataframe:" << endl;
    tests_data.run_all();
}