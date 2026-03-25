#include <cmath>
#include <algorithm>
#include <unordered_map>
#include "TestSuite.hpp"
#include "Data/Data.hpp"
#include "Preprocessing/Preprocessing.hpp"
#include "Models/Supervised/Classification/Classifications.hpp"

using namespace std;

// Testing Logistic Regression with corresponding stats
void LogReg(const Dataframe& x, const Dataframe& y, double penality, 
    const vector<double> clean_res0, const vector<double>& clean_res1) {

    // Through implemented code
    Class::LogisticRegression New_class(8.0, penality);
    New_class.fit(x, y);
    vector<double> to_test0 = New_class.get_stats();

    // Extract data from CoeffStats
    Class::CoeffStats inter = New_class.get_coefficient_stats()[1];
    vector<double> to_test1(clean_res1.size());
    for (size_t i = 0; i < clean_res1.size()/2; i++) {
        to_test1[i*2] = inter.beta[i];
        to_test1[(i*2)+1] = inter.p_value[i];
    }

    ASSERT_VEC_EPS(to_test0, clean_res0, 1e-2)

    ASSERT_VEC_EPS(to_test1, clean_res1, 1e-2)
}

void tests_Class() {
    

    // Initialization of our data 
    Dataframe iris = CsvHandler::loadCsv("../tests/datasets/classification/iris.csv", ',', true);
    Dataframe y = iris.transfer_col("target");
    iris.pop(0);
    
    for (size_t i = 0; i < iris.get_cols(); i++) {
        Scaling::scaling(iris, i);
    }

    // Through python
    vector<double> clean_res0 = {
        -10.8936,  // LL
        41.7873,   // AIC
        71.8936,   // BIC
        0.9339,    // McFadden
        0,         // Chi2 p-value
        0.9603,    // MCC
        0.9733,    // Accuracy
        0.9738,    // Precision
        0.9733,    // Recall
        0.9867,    // Specificity
        0.9733,    // F1
        0.9993     // ROC AUC
    };

    // Class 1 vs Class 2
    vector<double> clean_res1 = {
        6.3625,   // Beta0
        0.000012, // Beta0 p-value
        0.5379,   // Beta1
        0.575,    // Beta1 p-value
        0.9155,   // Beta2
        0.1879,   // Beta2 p-value
        -4.4588,  // Beta3
        0.0598,   // Beta3 p-value
        -5.6969,  // Beta4
        0.0031,   // Beta4 p-value
    };

    vector<double> clean_res2 = {
        -71.5451,  // LL
        163.0902,  // AIC
        193.1965,  // BIC
        0.5658,    // McFadden
        0,         // Chi2 p-value
        0.8637,    // MCC
        0.9067,    // Accuracy
        0.9130,    // Precision
        0.9067,    // Recall
        0.9533,    // Specificity
        0.9061,    // F1
        0.9735     // ROC AUC
    };

    // Class 1 vs Class 2
    vector<double> clean_res3 = {
        0.3718,    // Beta0
        0.2135,    // Beta0 p-value
        -0.1856,   // Beta1
        0.7273,    // Beta1 p-value
        -0.3076,   // Beta2
        0.3565,    // Beta2 p-value
        -0.3665,   // Beta3
        0.7502,    // Beta3 p-value
        -0.5387,   // Beta4
        0.4924,    // Beta4 p-value
    };

    // Add tests
    TestSuite::Tests tests_Class;

    tests_Class.add_test(
        bind(LogReg, iris, y, 0, clean_res0, clean_res1), 
        "Logistic Regression without penality"
    );

    tests_Class.add_test(
        bind(LogReg, iris, y, 1.5, clean_res2, clean_res3), 
        "Logistic Regression with Elastic Net penality"
    );

    cout << "Testing Classifications methods:" << endl;
    tests_Class.run_all();
}