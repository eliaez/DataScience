#include <cmath>
#include "Data/Data.hpp"
#include "Stats/stats_reg.hpp"
#include "Stats/Time_series.hpp"
#include "Models/Supervised/Regression/LinearReg.hpp"

namespace Stats_TS {











bool ADF_test(const std::vector<double>& y) {

    // Schwert rule
    size_t n = y.size();
    int p = (int)(12 * std::pow(n / 100.0, 0.25));
    int nb_cols = 3 + p;

    // Calculate our lags
    std::vector<double> delta_y;
    delta_y.reserve(n-1);
    for (size_t i = 1; i < n; i++) {
        delta_y.push_back(y[i] - y[i-1]);
    }

    // Create our matrix
    std::vector<double> y_v;
    std::vector<double> delta_x;
    y_v.reserve((n - p - 1));
    delta_x.reserve((n - p - 1) * nb_cols);
    for (size_t i = 0; i < nb_cols; i++) {
        for (size_t j = 0; j < (n - p - 1); j++) {

            if (i == 0) delta_x.push_back(1.0);
            else if (i == 1) delta_x.push_back(j + (p + 2));
            else if (i == 2) delta_x.push_back(y[j + p]);
            else delta_x.push_back(delta_y[j + (p + 2) - i]);

            if (i == (nb_cols - 1)) {
                y_v.push_back(delta_y[p + j]);
            }
        }
    }
    Dataframe X = {(n - p - 1), nb_cols, false, std::move(delta_x)};
    Dataframe Y = {(n - p - 1), 1, false, std::move(y_v)};
    
    // Linear Reg
    Reg::LinearRegression LinReg; 
    auto [X_const, XtXinv] = LinReg.fit_without_stats(X, Y);

    // Predict and residuals
    std::vector<double> y_pred = LinReg.predict(X);
    std::vector<double> residuals = Stats::get_residuals(Y.get_data(), y_pred);
    
    // Getting data 
    double gamma = LinReg.get_coeffs()[2];
    double se_gamma = Stats::OLS::stderr_b(residuals, XtXinv)[2];
    double adf_stat = gamma / se_gamma;
    double cv = critical_value_MacKinon(n);

    return adf_stat < cv;
}

double critical_value_MacKinon(size_t n) {
    double b0 = -2.8621, b1 = -2.738, b2 = -3.394;
    return b0 + b1 / n + b2 / (n * n);
}
}