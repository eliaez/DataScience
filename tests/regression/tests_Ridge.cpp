#include <cmath>
#include <algorithm>
#include <unordered_map>
#include "TestSuite.hpp"
#include "Data/Data.hpp"
#include "Validation/Validation.hpp"
#include "Preprocessing/Preprocessing.hpp"
#include "Models/Supervised/Regression/Regressions.hpp"

using namespace std;

// Testing Ridge Regression with Z-score Scaling for lambda = 1.0
void RidgeReg(Dataframe& x, const Dataframe& y, const vector<double> clean_res0,
    const vector<double>& clean_res1) {

    for (size_t i = 0; i < x.get_cols(); i++) {
        Scaling::scaling(x, i);
    }

    // Through implemented code
    Reg::RidgeRegression New_reg(1.0);
    New_reg.fit(x, y);
    vector<double> to_test0 = New_reg.get_stats();
    to_test0.erase(to_test0.begin()+9, to_test0.begin()+15);

    // Extract data from CoeffStats
    vector<Reg::CoeffStats> inter = New_reg.get_coefficient_stats();
    vector<double> to_test1(clean_res1.size());
    for (size_t i = 0; i < clean_res1.size(); i++) {
        to_test1[i] = inter[i].beta;
    }

    ASSERT_VEC_EPS(to_test0, clean_res0, 2e-2)

    ASSERT_VEC_EPS(to_test1, clean_res1, 2e-2)
}

void tests_Ridge() {
    

    // Initialization of our data 
    Dataframe california = CsvHandler::loadCsv("../tests/datasets/california/california_housing.csv", ',', true);
    Dataframe y = california.transfer_col("PRICE");

    // Through python
    vector<double> clean_res0 = {
        0.6062,     // R2
        0.6061,     // R2 adjusted
        8.00,       // Effective DF
        0.5243,     // MSE
        0.7241,     // RMSE
        0.5312,     // MAE
        45263.54,   // AIC
        45327.00,   // BIC
        0.5574,     // Durbin-Watson - rho value
    };

    vector<double> clean_res1 = {
        2.0686,       // Beta0
        0.8296,       // Beta1
        0.1188,       // Beta2
        -0.2654,      // Beta3
        0.3055,       // Beta4
        -0.0045,      // Beta5
        -0.0393,      // Beta6
        -0.8993,      // Beta7
        -0.8699,      // Beta8
    };

    // Add tests
    TestSuite::Tests tests_Ridge;

    tests_Ridge.add_test(
        bind(RidgeReg, california, y, clean_res0, clean_res1), 
        "Ridge Regression with Z-score Scaling for lambda = 1.0"
    );

    /*tests_Ridge.add_test(
        bind(LinearReg, california, y, clean_res2, clean_res3), 
        "Linear Regression - cov_type = classical"
    );*/

    cout << "Testing Ridge, Stats, Preprocessing and Validation functions:" << endl;
    tests_Ridge.run_all();
}