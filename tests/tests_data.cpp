#include "Utils/TestSuite.hpp"
#include "Data/Data.hpp"

using namespace std;

// Testing data after change_layout
void changelayout(Dataframe df1, const std::vector<double>& res, const std::string& backend) {
    
    Dataframe df1_t = df1.change_layout(backend);

    ASSERT_EQ(df1_t.get_data(), res)
}

// Testing the transfer of a column from a Df to another new one
void transfercol_t(Dataframe& iris, const string& col,
    const Dataframe& iris_x, const std::vector<double>& iris_y) {

    Dataframe y = iris.transfer_col(col);
    
    // Data new df
    ASSERT_EQ(y.get_data(), iris_y)

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
    Dataframe iris_t = CsvHandler::loadCsv("../tests/datasets/iris_t.csv", ',', false);
    Dataframe iris_x = CsvHandler::loadCsv("../tests/datasets/iris_x.csv");
    Dataframe iris_y = CsvHandler::loadCsv("../tests/datasets/iris_y.csv");

    Dataframe mat = CsvHandler::loadCsv("../tests/datasets/mat.csv", ',', false);
    Dataframe mat_t = CsvHandler::loadCsv("../tests/datasets/mat_t.csv", ',', false);

    // Add tests
    TestSuite::Tests tests_data;

    tests_data.add_test(
        bind(changelayout, iris, iris_t.get_data(), "Naive"), 
        "Change layout Naive"
    );

    #ifdef __AVX2__
        tests_data.add_test(
            bind(changelayout, iris, iris_t.get_data(), "AVX2"), 
            "Change layout AVX2 v1"
        );

        tests_data.add_test(
            bind(changelayout, mat, mat_t.get_data(), "AVX2"), 
            "Change layout AVX2 v2"
        );
    #endif

    tests_data.add_test(
        bind(changelayout, iris, iris_t.get_data(), "Eigen"), 
        "Change layout Eigen v1"
    );

    tests_data.add_test(
        bind(changelayout, mat, mat_t.get_data(), "Eigen"), 
        "Change layout Eigen v2"
    );

    #ifdef USE_MKL
        tests_data.add_test(
            bind(changelayout, iris, iris_t.get_data(), "MKL"), 
            "Change layout MKL v1"
        );

        tests_data.add_test(
            bind(changelayout, mat, mat_t.get_data(), "MKL"), 
            "Change layout MKL v2"
        );
    #endif

    tests_data.add_test(
        bind(transfercol_t, iris, "target", iris_x, iris_y.get_data()), 
        "Transfer column"
    );

    cout << "Testing Dataframe:" << endl;
    tests_data.run_all();
}