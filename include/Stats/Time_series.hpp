#pragma once 

#include <vector>

namespace Stats_TS {
    class ARIMA {
        private:
            int p_;
            int d_;
            int q_;
            int s_;
            bool seasonal_;

        public:
            // Set params to -1 for them to be automatically detected
            ARIMA(int p = -1, int d = -1, int q = -1, bool seasonal = true, int s = -1) :
                p_(p), d_(d), q_(q), seasonal_(seasonal), s_(s) {};

            // Function to detect and returns ARIMA params on your vector (eq to fit)
            void fit(const std::vector<double>& col);

            // Function to predict based on your ARIMA params
            std::vector<double> predict(const std::vector<double>& col);

            // Function to fir and predict based on your ARIMA params on your vector
            std::vector<double> fit_predict(const std::vector<double>& col);

            // Setter
            void set_seasonal(bool seasonal) { seasonal_ = seasonal; }

            // Getters
            // Returns p, d, q, s and seasonal
            std::vector<int> get_params() { return {p_, d_, q_, s_, seasonal_ ? 1 : 0}; }
    };

    // PACF through Durbin-Levinson, will return p
    int Pacf(const std::vector<double>& y);

    // Augmented Dickey-Fuller test with y, will return adf_stat
    double ADF_test(const std::vector<double>& y);

    // Necessary for ADF
    double critical_value_MacKinon(size_t n);
}



