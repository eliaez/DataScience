#include "Utils/TestSuite.hpp"
#include "Data/Data.hpp"

using namespace std;

// Testing data in col-major with loadingCsv
void loadCsv_t(const string& filepath, const vector<double>& iris_colmajor) {

    Dataframe iris = CsvHandler::loadCsv(filepath);

    ASSERT_EQ(iris.get_data(), iris_colmajor)
}

// Testing data after change_layout
void changelayout_t(Dataframe iris, const vector<double>& iris_rowmajor) {
    
    iris.change_layout_inplace();

    ASSERT_EQ(iris.get_data(), iris_rowmajor)
}

// Testing the transfer of a column from a Df to another new one
void transfercol_t(Dataframe& iris, const string& col,
    const Dataframe& iris_bis, const vector<double>& res_y) {

    Dataframe y = iris.transfer_col(col);
    
    // Data new df
    ASSERT_EQ(y.get_data(), res_y)

    // Headers new df
    ASSERT_EQ(y.get_headers()[0], col)

    // Data former df
    ASSERT_EQ(iris.get_data(), iris_bis.get_data())

    // Headers former df
    ASSERT_EQ(iris.get_headers(), iris_bis.get_headers())
}

void tests_data() {

    // Initialization of our data 
    size_t m = 3, n = 4;

    string col = "sepal width (cm)";

    vector<string> iris_headers = {"sepal length (cm)", "sepal width (cm)",
        "petal length (cm)", "petal width (cm)", "target"};

    vector<string> iris_headers_bis = {"sepal length (cm)", "petal length (cm)", 
        "petal width (cm)", "target"};

    vector<double> res_y = {3.5,3.0,3.2};

    vector<double> iris_colmajor = {5.1,4.9,4.7,3.5,3.0,3.2,1.4,1.4,1.3,0.2,0.2,0.2};
    vector<double> iris_colmajor_bis = {5.1,4.9,4.7,1.4,1.4,1.3,0.2,0.2,0.2};

    vector<double> iris_rowmajor = {5.1,3.5,1.4,0.2,4.9,3.0,1.4,0.2,4.7,3.2,1.3,0.2};

    Dataframe iris = {m, n, false, iris_colmajor, iris_headers};
    Dataframe iris_bis = {m, n-1, false, iris_colmajor_bis, iris_headers_bis};

    // Add tests
    TestSuite::Tests tests_data;

    tests_data.add_test(
        bind(loadCsv_t, "../tests/iris.csv", iris_colmajor), 
        "Loading Csv"
    );

    tests_data.add_test(
        bind(changelayout_t, iris, iris_rowmajor), 
        "Change layout"
    );

    tests_data.add_test(
        bind(transfercol_t, iris, col, iris_bis, res_y), 
        "Transfer column"
    );

    cout << "Testing Dataframe:" << endl;
    tests_data.run_all();
}