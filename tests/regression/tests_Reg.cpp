#include <cmath>
#include <algorithm>
#include <unordered_map>
#include "TestSuite.hpp"
#include "Data/Data.hpp"
#include "Validation/Validation.hpp"
#include "Preprocessing/Preprocessing.hpp"
#include "Models/Supervised/Regression/Regressions.hpp"

using namespace std;

// Testing Other Regression with Z-score Scaling for lambda = 1.0
void OtherReg(Dataframe& x, const Dataframe& y, const vector<double> clean_res0,
    const vector<double>& clean_res1, const std::string& model) {

    for (size_t i = 0; i < x.get_cols(); i++) {
        Scaling::scaling(x, i);
    }

    // Through implemented code
    std::unique_ptr<Reg::RegressionBase> New_reg;
    if (model == "Ridge") {
        New_reg = std::make_unique<Reg::RidgeRegression>(1.0);
    }
    else if (model == "Lasso") {
        New_reg = std::make_unique<Reg::LassoRegression>(0.1);
    }
    else if (model == "Elastic") {
        New_reg = std::make_unique<Reg::ElasticRegression>(0.1, 0.5);
    }
    
    New_reg->fit(x, y);
    vector<double> to_test0 = New_reg->get_stats();
    to_test0.erase(to_test0.begin()+9, to_test0.begin()+15);

    // Extract data from CoeffStats
    vector<Reg::CoeffStats> inter = New_reg->get_coefficient_stats();
    vector<double> to_test1(clean_res1.size());
    for (size_t i = 0; i < clean_res1.size(); i++) {
        to_test1[i] = inter[i].beta;
    }

    ASSERT_VEC_EPS(to_test0, clean_res0, 2e-2)

    ASSERT_VEC_EPS(to_test1, clean_res1, 2e-2)
}

void tests_Reg() {
    

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

    vector<double> clean_res2 = {
        0.4937,     // R2
        0.4935,     // R2 adjusted
        3.00,       // Effective DF
        0.6741,     // MSE
        0.8211,     // RMSE
        0.6203,     // MAE
        50441.12,   // AIC
        50464.92,   // BIC
        0.6871,     // Durbin-Watson - rho value
    };

    vector<double> clean_res3 = {
        2.0686,       // Beta0
        0.7057,       // Beta1
        0.1060,       // Beta2
        0,            // Beta3
        0,            // Beta4
        0,            // Beta5
        0,            // Beta6
        -0.0112,      // Beta7
        0,            // Beta8
    };

    vector<double> clean_res4 = {
        0.5272,     // R2
        0.5270,     // R2 adjusted
        4.0,       // Effective DF
        0.6295,     // MSE
        0.7934,     // RMSE
        0.5953,     // MAE
        49030.21,   // AIC
        49061.95,   // BIC
        0.6591,     // Durbin-Watson - rho value
    };

    vector<double> clean_res5 = {
        2.0686,       // Beta0
        0.7088,       // Beta1
        0.1370,       // Beta2
        0,            // Beta3
        0,            // Beta4
        0,            // Beta5
        0,            // Beta6
        -0.1757,      // Beta7
        -0.1333,      // Beta8
    };

    // Add tests
    TestSuite::Tests tests_Reg;

    tests_Reg.add_test(
        bind(OtherReg, california, y, clean_res0, clean_res1, "Ridge"), 
        "Ridge Regression with Z-score Scaling for lambda = 1.0"
    );

    tests_Reg.add_test(
        bind(OtherReg, california, y, clean_res2, clean_res3, "Lasso"), 
        "Lasso Regression with Z-score Scaling for lambda = 0.1"
    );

    tests_Reg.add_test(
        bind(OtherReg, california, y, clean_res4, clean_res5, "Elastic"), 
        "ElasticNet Regression with Z-score Scaling for alpha = 0.1, l1_ratio = 0.5"
    );

    cout << "Testing Ridge, Stats, Preprocessing functions:" << endl;
    tests_Reg.run_all();
}