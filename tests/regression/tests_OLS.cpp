#include <cmath>
#include <algorithm>
#include <unordered_map>
#include "TestSuite.hpp"
#include "Data/Data.hpp"
#include "Models/Regressions.hpp"

using namespace std;

vector<int> create_clusters(const vector<double>& latitude) {
    vector<int> clusters(latitude.size());
    unordered_map<int, int> mapping;
    int next_id = 0;
    
    for(size_t i = 0; i < latitude.size(); ++i) {
        int rounded = static_cast<int>(round(latitude[i]));
        
        if(mapping.find(rounded) == mapping.end()) {
            mapping[rounded] = next_id++;
        }
        
        clusters[i] = mapping[rounded];
    }
    
    return clusters;
}

// Testing Linear Regression with corresponding stats
void LinearReg(const Dataframe& x, const Dataframe& y, const vector<double> clean_res0,
    const vector<double>& clean_res1) {

    // Through implemented code
    Reg::LinearRegression New_reg;
    New_reg.fit(x, y);
    vector<double> to_test0 = New_reg.get_stats();
    to_test0.erase(to_test0.begin()+9, to_test0.begin()+15);

    // Extract data from CoeffStats
    vector<Reg::CoeffStats> inter = New_reg.get_coefficient_stats();
    vector<double> to_test1(clean_res1.size());
    for (size_t i = 0; i < (clean_res1.size()/2); i++) {
        to_test1[i*2] = inter[i].beta;
        to_test1[(i*2)+1] = inter[i].t_stat;
    }

    ASSERT_VEC_EPS(to_test0, clean_res0, 2e-2)

    ASSERT_VEC_EPS(to_test1, clean_res1, 2e-2)
}

// Testing Linear Regression with specific cov_type 
void LinearRegCovtype(const Dataframe& x, const Dataframe& y, const string& cov_type,
    const vector<double> clean_res0, const vector<double>& clean_res1, const vector<int>& cluster_ids) {

    // Through implemented code
    Reg::LinearRegression New_reg(cov_type, cluster_ids, {});
    New_reg.fit(x, y);
    vector<double> inter0 = New_reg.get_stats();
    vector<double> to_test0 = {inter0[0], inter0[1], inter0[5]};

    // Extract data from CoeffStats
    vector<Reg::CoeffStats> inter = New_reg.get_coefficient_stats();
    vector<double> to_test1(clean_res1.size());
    for (size_t i = 0; i < clean_res1.size(); i++) {
        to_test1[i] = inter[i].t_stat;
    }

    ASSERT_VEC_EPS(to_test0, clean_res0, 1e-4)

    ASSERT_VEC_EPS(to_test1, clean_res1, 1e-4)
}

void tests_OLS() {
    

    // Initialization of our data 
    Dataframe california = CsvHandler::loadCsv("../tests/datasets/california/california_housing.csv", ',', true);
    Dataframe x = california["MedInc"];
    Dataframe y = california.transfer_col("PRICE");

    // Through python
    vector<double> clean_res0 = {
        0.473447,   // R2
        -1.0,       // R2 adjusted
        0.701131,   // MSE
        0.837335,   // RMSE
        0.636359,   // MAE
        18556.57,   // Fisher - F
        0.0,        // Fisher - p-value
        0.6727,     // Durbin-Watson - rho value
        0.0         // Breusch-Pagan - p-value
    };

    vector<double> clean_res1 = {
        0.450855,   // Beta0
        34.08159,   // Beta0 t-value
        0.417938,   // Beta1
        136.2225    // Beta1 t-value
    };

    vector<double> clean_res2 = {
        0.606232,   // R2
        0.606079,   // R2 adjusted
        0.524320,   // MSE
        0.724100,   // RMSE
        0.531164,   // MAE
        3970.360,   // Fisher - F
        0.0,        // Fisher - p-value
        0.5574,     // Durbin-Watson - rho value
        0.0,        // Breusch-Pagan - p-value
        NAN,
        2.501295,   // VIF MedInc
        1.241254,   // VIF HouseAge
        8.342786,   // VIF ...
        6.994995,   // VIF ...
        1.138125,   // VIF ...
        1.008324,   // VIF ...
        9.297624,   // VIF ...
        8.962263,   // VIF ...
    };

    vector<double> clean_res3 = {
        -36.9419,       // Beta0
        -56.0665,       // Beta0 t-value
        0.436693,       // Beta1
        104.0538,       // Beta1 t-value
        0.009435,       // Beta2
        21.1432,        // Beta2 t-value
        -0.107322,      // Beta3
        -18.2354,       // Beta3 t-value
        0.645065,       // Beta4
        22.9276,        // Beta4 t-value
        -0.0000039,     // Beta5
        -0.8373,        // Beta5 t-value
        -0.003786,      // Beta6
        -7.7686,        // Beta6 t-value
        -0.421314,      // Beta7
        -58.5414,       // Beta7 t-value
        -0.434513,      // Beta8
        -57.6822        // Beta8 t-value
    };

    vector<double> clean_res4 = {
        0.606232,   // R2
        0.606079,   // R2 adjusted
        2967.892,   // Fisher - F
    };

    vector<double> clean_res5 = {
        -42.1974,       // Beta0 t-value
        39.3968,        // Beta1 t-value
        17.5394,        // Beta2 t-value
        -5.9371,        // Beta3 t-value
        4.6153,         // Beta4 t-value
        -0.8508,        // Beta5 t-value
        -1.7336,        // Beta6 t-value
        -44.2978,       // Beta7 t-value
        -44.5273        // Beta8 t-value
    };

    vector<double> clean_res6 = {
        0.606232,   // R2
        0.606079,   // R2 adjusted
        801.6303,   // Fisher - F
    };

    vector<double> clean_res7 = {
        -20.8101,       // Beta0 t-value
        34.9991,        // Beta1 t-value
        9.6466,         // Beta2 t-value
        -5.5468,        // Beta3 t-value
        5.4663,         // Beta4 t-value
        -0.6747,        // Beta5 t-value
        -3.9143,        // Beta6 t-value
        -22.0123,       // Beta7 t-value
        -21.5875        // Beta8 t-value
    };

    vector<double> clean_res8 = {
        0.606232,   // R2
        0.606079,   // R2 adjusted
        2949.453,   // Fisher - F
    };

    vector<double> clean_res9 = {
        -6.8096,        // Beta0 t-value
        21.4457,        // Beta1 t-value
        4.3902,         // Beta2 t-value
        -2.8132,        // Beta3 t-value
        3.3108,         // Beta4 t-value
        -0.3380,        // Beta5 t-value
        -3.0115,        // Beta6 t-value
        -6.0274,        // Beta7 t-value
        -6.5804         // Beta8 t-value
    };

    vector<int> empty_vec = {};
    vector<double> latitude = california["Latitude"].get_data(); 
    vector<int> clusters = create_clusters(latitude);

    // Add tests
    TestSuite::Tests tests_OLS;

    tests_OLS.add_test(
        bind(LinearReg, x, y, clean_res0, clean_res1), 
        "Simple Linear Regression"
    );

    tests_OLS.add_test(
        bind(LinearReg, california, y, clean_res2, clean_res3), 
        "Linear Regression - cov_type = classical"
    );

    tests_OLS.add_test(
        bind(LinearRegCovtype, california, y, "HC3", clean_res4, clean_res5, empty_vec), 
        "Linear Regression - cov_type = HC3"
    );

    tests_OLS.add_test(
        bind(LinearRegCovtype, california, y, "HAC", clean_res6, clean_res7, empty_vec), 
        "Linear Regression - cov_type = HAC"
    );

    tests_OLS.add_test(
        bind(LinearRegCovtype, california, y, "cluster", clean_res8, clean_res9, clusters), 
        "Linear Regression - cov_type = cluster"
    );

    cout << "Testing OLS and Stats functions:" << endl;
    tests_OLS.run_all();
}