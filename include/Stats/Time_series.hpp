#pragma once 

#include <vector>

namespace Stats_TS {;

    // Function to detect and returns ARIMA params on your vector
    // Will return p, d, q
    std::vector<int> detect_ARIMA(const std::vector<double>& col);

    // Function to detect and returns ARIMA params on your vector
    // Will return p, d, q, P, D, Q, s and seasonality
    std::vector<int> detect_SARIMA(const std::vector<double>& col);

    // FFT (Fast Fourier Transformation)
    int Fft(const std::vector<double>& y);

    // Seasonality test Kruskal_Wallis
    bool Kruskal_Wallis(const std::vector<double>& y, int s);

    // Linear detrend for Kurskal_Wallis
    std::vector<double> linear_detrend(const std::vector<double>& y);

    // ACF to find period (seasonality), will return s
    int Acf_s(const std::vector<double>& y);

    // ACF_s will return Q
    int Acf_s_q(const std::vector<double>& y, int s);

    // ACF, will return q (MA)
    int Acf(const std::vector<double>& y);

    // PACF through Durbin-Levinson, will return p (AR)
    int Pacf(const std::vector<double>& y);

    // PACF_s through Durbin-Levinson, will return P
    int Pacf_s(const std::vector<double>& y, int s);

    // Augmented Dickey-Fuller test with y, will return adf_stat for stationarity
    double ADF_test(const std::vector<double>& y);

    // Necessary for ADF
    double critical_value_MacKinon(size_t n);
}



