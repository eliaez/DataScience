#include <cmath>
#include "Data/Data.hpp"
#include "Stats/stats_reg.hpp"
#include "Stats/Time_series.hpp"
#include "Models/Supervised/Regression/LinearReg.hpp"

namespace Stats_TS {

void ARIMA::fit(const std::vector<double>& col) {

    // Detect d
    int d = 0;
    bool keep_cond = true;
    std::vector<double> y = col;
    while (keep_cond && d < 3) {

        // Diff
        if (d > 0) {
            std::vector<double> diff;
            diff.reserve(y.size() - 1);
            for (size_t i = 1; i < y.size(); i++) {
                diff.push_back(y[i] - y[i-1]);
            }   
            y = diff;
        }

        // ADF test
        double adf_stat = ADF_test(y);
        double cv = critical_value_MacKinon(y.size());
        if (adf_stat < cv) keep_cond = false;
        else d++;
    }
    d_ = d;

    // Detect p (AR)
    p_ = Pacf(y);

    // Detect q (MA)

}

int Pacf(const std::vector<double>& y) {

    size_t n = y.size();
    double mean_y = Stats::mean(y);
    size_t kmax = std::sqrt(n) > 40 ? 40 : std::sqrt(n);
    std::vector<std::vector<double>> phi(kmax + 1, std::vector<double>(kmax + 1, 0.0));

    // Calculate auto-covar for all k
    double gamma_0;
    std::vector<double> rho(kmax + 1, 0.0);
    for (size_t k = 0; k <= kmax; k++) {
        
        double gamma = 0.0;
        for (size_t i = k; i < n; i++) {
            gamma += (y[i] - mean_y) * (y[i-k] - mean_y);
        }   
        gamma /= n;
        if (k == 0) gamma_0 = gamma;
        rho[k] = gamma / gamma_0;
    }

    // Durbin-Levinson
    int p = 0;
    phi[1][1] = rho[1];
    double seuil = 1.96 / std::sqrt(n);

    // Test 1
    if (std::abs(phi[1][1]) > seuil) p = 1;

    for (size_t k = 2; k <= kmax; k++) {

        // Calculate phi_k,k
        double sum0 = 0.0, sum1 = 0.0;
        for (size_t j = 1; j <= k-1; j++) {
            sum0 += phi[k-1][j] * rho[k-j];
            sum1 += phi[k-1][j] * rho[j];
        }
        phi[k][k] = (rho[k] - sum0) / (1 - sum1);

        // Calculate intermediary values
        for (size_t j = 1; j <= k-1; j++) {
            phi[k][j] = phi[k-1][j] - phi[k][k] * phi[k-1][k-j];
        }

        // Test
        if (std::abs(phi[k][k]) > seuil) p = k;
    }
    return p;
}

double ADF_test(const std::vector<double>& y) {

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

    return adf_stat;
}

double critical_value_MacKinon(size_t n) {
    double b0 = -2.8621, b1 = -2.738, b2 = -3.394;
    return b0 + b1 / n + b2 / (n * n);
}
}