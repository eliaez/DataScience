#pragma once 

#include <vector>

namespace Stats_TS {
    class ARIMA {
        protected:
            int p_;
            int d_;
            int q_;
            std::vector<double> y_diff;
            std::vector<double> ar_coeffs;
            std::vector<double> ma_coeffs;
            std::vector<double> residuals;

        public:
            // Set params to -1 for them to be automatically detected
            ARIMA(int p = -1, int d = -1, int q = -1) :
                p_(p), d_(d), q_(q) {};

            // Function to detect and returns ARIMA params on your vector
            // Will return p, d, q
            virtual std::vector<int> detect(const std::vector<double>& col);

            // Function to detect (if params still at -1) and fit ARIMA params on your vector
            virtual void fit(const std::vector<double>& col);

            // Function to predict based on your ARIMA params
            virtual std::vector<double> predict(const std::vector<double>& col);

            // Function to fir and predict based on your ARIMA params on your vector
            virtual std::vector<double> fit_predict(const std::vector<double>& col);

            // Getters
            // Returns p, d, q
            virtual std::vector<int> get_params() const { return {p_, d_, q_}; }
            virtual const std::vector<double>& get_ydiff() const { return y_diff; }
            virtual const std::vector<double>& get_arcoeffs() const { return ar_coeffs; }
            virtual const std::vector<double>& get_arcoeffs() const { return ma_coeffs; }
            virtual const std::vector<double>& get_residuals() const { return residuals; }
    };

    class SARIMA : public ARIMA {
        private:
            int P_;
            int D_;
            int Q_;
            int s_;
            bool seasonal_;

        public:
            // Set params to -1 for them to be automatically detected
            SARIMA(int p = -1, int d = -1, int q = -1, int P = -1, int D = -1, int Q = -1, int s = -1) :
                ARIMA(p, d, q), P_(P), D_(D), Q_(Q), s_(s) {};

            // Function to detect and returns SARIMA params on your vector
            // Will return p, d, q, P, D, Q, s and seasonality 
            std::vector<int> detect(const std::vector<double>& col) override;

            // Function to detect and returns SARIMA params on your vector
            void fit(const std::vector<double>& col) override;

            // Function to predict based on your SARIMA params
            std::vector<double> predict(const std::vector<double>& col) override;

            // Function to fir and predict based on your SARIMA params on your vector
            std::vector<double> fit_predict(const std::vector<double>& col) override;

            // Setter
            void set_seasonal(bool seasonal) { seasonal_ = seasonal; }

            // Getters
            // Returns p, d, q, P, D, Q, s and seasonal
            std::vector<int> get_params() const override { return {p_, d_, q_, P_, D_, Q_, s_, seasonal_ ? 1 : 0}; }
    };

    // FFT (Fast Fourier Transformation)
    int Fft(const std::vector<double>& y);

    // Seasonality test Kruskal_Wallis
    bool Kruskal_Wallis(const std::vector<double>& y, int s);

    // ACF to find period (seasonality), will return s
    int Acf_s(const std::vector<double>& y);

    // ACF, will return q (MA)
    int Acf(const std::vector<double>& y);

    // PACF through Durbin-Levinson, will return p (AR)
    int Pacf(const std::vector<double>& y);

    // Augmented Dickey-Fuller test with y, will return adf_stat for stationarity
    double ADF_test(const std::vector<double>& y);

    // Necessary for ADF
    double critical_value_MacKinon(size_t n);
}



