#include <cmath>
#include <complex>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Stats/stats_reg.hpp"
#include "Stats/Time_series.hpp"
#include <boost/math/distributions/chi_squared.hpp>
#include "Models/Supervised/Regression/RidgeReg.hpp"

using namespace Utils;

namespace Stats_TS {

std::vector<int> detect_ARIMA(const std::vector<double>& col) {
    int p;
    int q;
    std::vector<double> y;

    // Detect d (stationarity)
    int d = 0;
    bool keep_cond = true;
    y = col;
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
        if (adf_stat < cv) {
            keep_cond = false;
        }
        else if (d == 2) {
            d = -2;
            break;
        }
        else d++;
    }
    if (d == -2) {
        p = -2;
        q = -2;
        return {p, d, q};
    }

    // Detect p (AR)
    p = Pacf(y);

    // Detect q (MA)
    q = Acf(y);

    return {p, d, q};
}

std::vector<int> detect_SARIMA(const std::vector<double>& col) {

    bool seasonal = true;
    std::vector<double> y = col;
    std::vector<double> y_det = linear_detrend(y);
    
    // Detect a period with ACF and first approx of period s
    int s = Acf_s(y_det);

    // Testing if we have or not seasonality
    if (Kruskal_Wallis(y_det, s) && s != 0) {
        s = Fft(y_det);
    }
    else {
        s = 0;
        seasonal = false;
    }

    if (!seasonal) {
        // Fallback sur ARIMA classique
        std::vector<int> arima = detect_ARIMA(col);
        return {arima[0], arima[1], arima[2], 0, 0, 0, 0, 0};
    }

    // Detect D
    int D = 0;
    bool keep_cond = true;
    while (keep_cond && D < 2) {

        if (D > 0) {

            std::vector<double> diff;
            diff.reserve(y.size() - s);
            for (size_t i = s; i < y.size(); i++) {
                diff.push_back(y[i] - y[i - s]);
            }
            y = diff;
        }

        double adf_stat = ADF_test(y);
        double cv = critical_value_MacKinon(y.size());
        if (adf_stat < cv) {
            keep_cond = false;
        } else {
            D++;
        }
    }

    // Detect P and Q
    int P = Pacf_s(y, s);
    int Q = Acf_s_q(y, s);

    // Detect p, d, q on our y
    std::vector<int> arima = detect_ARIMA(y);
    int p = arima[0], d = arima[1], q = arima[2];

    // (p,d,q)(P,D,Q,s,seasonal)
    return {p, d, q, P, D, Q, s, seasonal ? 1 : 0};
}

int Fft(const std::vector<double>& y) {

    // Init
    size_t n = y.size();
    size_t nb_val = std::floor(n / 2);
    std::vector<double> P;
    P.reserve(nb_val);

    // Calculate P_f
    double pi = acos(-1.0);
    std::complex<double> imag(0, 1);
    for (size_t i = 0; i < nb_val; i++) {
        
        double f = static_cast<double>(i) / n;
        std::complex<double> x = 0.0;
        for (size_t j = 0; j < n; j++) {
            x += y[j] * std::exp(-2.0 * pi * imag * f * static_cast<double>(j));
        }
        P.push_back(std::norm(x));
    }

    // Getting idx of max element
    size_t it = std::max_element(P.begin() + 1, P.end()) - P.begin();

    double freq = static_cast<double>(it) / n;
    return static_cast<int>(std::round(1.0 / freq));
}

bool Kruskal_Wallis(const std::vector<double>& y, int s) {

    size_t n = y.size();

    // Get ranks
    std::vector<double> ranks = compute_ranks(y);
    
    // Sum ranks per group and count elements
    std::vector<double> R(s, 0.0);
    std::vector<int> n_i(s, 0);
    
    for (size_t i = 0; i < n; i++) {
        int group = i % s;
        R[group] += ranks[i];
        n_i[group]++;
    }

    // Calculating H 
    double H = 0.0;
    for (size_t i = 0; i < s; i++) H += (R[i] * R[i]) / n_i[i];
    H *= 12.0 / (n * (n + 1));
    H -= 3 * (n + 1);

    if (s <= 1) return false;
    boost::math::chi_squared dist(s - 1);
    double p_value = 1.0 - boost::math::cdf(dist, H);
    return p_value < 0.05;
}

int Acf_s_q(const std::vector<double>& y, int s) {

    size_t n = y.size();
    double mean_y = Stats::mean(y);
    size_t kmax = n / 4 > 40 ? 40 : n / 4;
    size_t Qmax = kmax / s;
    if (Qmax < 1) return 0;

    double gamma_0;
    double threshold = 1.96 / std::sqrt(n);
    int Q = 0;

    for (size_t k = 0; k <= Qmax; k++) {

        size_t lag = k * s;
        double gamma = 0.0;
        for (size_t i = lag; i < n; i++) {
            gamma += (y[i] - mean_y) * (y[i - lag] - mean_y);
        }
        gamma /= n;
        if (k == 0) gamma_0 = gamma;

        double rho = gamma / gamma_0;

        if ((std::abs(rho) <= threshold) && (k > 0)) {
            Q = k - 1;
            break;
        }
    }
    return Q;
}

std::vector<double> linear_detrend(const std::vector<double>& y) {
    size_t n = y.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    
    for (size_t i = 0; i < n; i++) {
        sx  += i;
        sy  += y[i];
        sxx += i * i;
        sxy += i * y[i];
    }
    
    double c = n * sxx - sx * sx;
    double b = (n * sxy - sx * sy) / c;
    double a = (sy - b * sx) / n;
    
    std::vector<double> y_det(n);
    for (size_t i = 0; i < n; i++)
        y_det[i] = y[i] - (a + b * i);
    return y_det;
}

int Acf_s(const std::vector<double>& y) {

    size_t n = y.size();
    double mean_y = Stats::mean(y);
    size_t kmax = n / 2 > 40 ? 40 : n / 2;

    // Calculate auto-covar for all k
    double gamma_0;
    double threshold = 1.96 / std::sqrt(n);
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

    // Finding patterns within significant acf stats
    int s = 0;
    for (size_t i = 4; i <= kmax; i++) {

        // If significant value
        if (std::abs(rho[i]) > threshold) {
            if (rho[i-1] < rho[i] && rho[i+1] < rho[i]) {
                s = static_cast<int>(i);
                break;
            }
        }
    }
    return s;
}

int Acf(const std::vector<double>& y) {

    size_t n = y.size();
    double mean_y = Stats::mean(y);
    size_t kmax = n / 4 > 40 ? 40 : n / 4;

    // Calculate auto-covar for all k
    int q = 0;
    double rho;
    double gamma_0;
    double threshold = 1.96 / std::sqrt(n);
    for (size_t k = 0; k <= kmax; k++) {
        
        double gamma = 0.0;
        for (size_t i = k; i < n; i++) {
            gamma += (y[i] - mean_y) * (y[i-k] - mean_y);
        }   
        gamma /= n;
        if (k == 0) gamma_0 = gamma;
        rho = gamma / gamma_0;

        // Test
        if ((std::abs(rho) <= threshold) && (k > 0)) {
            q = k - 1;
            break;
        }
    }
    return q;
}

int Pacf(const std::vector<double>& y) {

    size_t n = y.size();
    double mean_y = Stats::mean(y);
    size_t kmax = n / 4 > 40 ? 40 :  n / 4;
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
    double threshold = 1.96 / std::sqrt(n);

    // Test 1
    if (std::abs(phi[1][1]) > threshold) p = 1;

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
        if (std::abs(phi[k][k]) > threshold) {
            p = k;
        } else {
            break;
        }
    }
    return p;
}

int Pacf_s(const std::vector<double>& y, int s) {

    size_t n = y.size();
    double mean_y = Stats::mean(y);
    size_t kmax = n / 4 > 40 ? 40 : n / 4;
    size_t Pmax = kmax / s;
    if (Pmax < 1) return 0;

    std::vector<std::vector<double>> phi(Pmax + 1, std::vector<double>(Pmax + 1, 0.0));

    // ACF aux lags multiples de s
    double gamma_0;
    std::vector<double> rho(Pmax + 1, 0.0);
    for (size_t k = 0; k <= Pmax; k++) {
        
        size_t lag = k * s;
        double gamma = 0.0;
        for (size_t i = lag; i < n; i++) {
            gamma += (y[i] - mean_y) * (y[i - lag] - mean_y);
        }
        gamma /= n;
        if (k == 0) gamma_0 = gamma;
        rho[k] = gamma / gamma_0;
    }

    // Durbin-Levinson sur ces lags
    int P = 0;
    phi[1][1] = rho[1];
    double threshold = 1.96 / std::sqrt(n);

    if (std::abs(phi[1][1]) > threshold) P = 1;

    for (size_t k = 2; k <= Pmax; k++) {

        double sum0 = 0.0, sum1 = 0.0;
        for (size_t j = 1; j <= k-1; j++) {
            sum0 += phi[k-1][j] * rho[k-j];
            sum1 += phi[k-1][j] * rho[j];
        }
        phi[k][k] = (rho[k] - sum0) / (1 - sum1);

        for (size_t j = 1; j <= k-1; j++) {
            phi[k][j] = phi[k-1][j] - phi[k][k] * phi[k-1][k-j];
        }

        if (std::abs(phi[k][k]) > threshold) {
            P = k;
        } else {
            break;
        }
    }
    return P;
}

double ADF_test(const std::vector<double>& y) {

    // Schwert rule
    size_t n = y.size();
    int p = static_cast<int>(12 * std::pow(n / 100.0, 0.25));
    size_t nb_cols = 3 + p;

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
    
    // Ridge Reg
    Reg::RidgeRegression RiReg(1e-10);
    auto [X_const, XtXinv] = RiReg.fit_without_stats(X, Y);

    // Predict and residuals
    std::vector<double> y_pred = RiReg.predict(X);
    std::vector<double> residuals = Stats::get_residuals(Y.get_data(), y_pred);
    
    // Getting data 
    double gamma = RiReg.get_coeffs()[3];
    double se_gamma = Stats::OLS::stderr_b(residuals, XtXinv)[2];
    double adf_stat = gamma / se_gamma;

    return adf_stat;
}

double critical_value_MacKinon(size_t n) {
    double b0 = -2.8621, b1 = -2.738, b2 = -3.394;
    return b0 + b1 / n + b2 / (n * n);
}
}