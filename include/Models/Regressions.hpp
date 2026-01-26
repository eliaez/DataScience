#pragma once

#include <vector>



namespace Reg {
    class LinearRegression {
        private:
            double intercept;
            double slope;
            std::vector<double> coefficients;
            std::vector<double> stats; // May change
            bool is_fitted;

        public:
            // Constructor 
            LinearRegression() : is_fitted(false) {}; // Init to get col major or warn user 

            // Training
            void fit(const std::vector<double>& x, const std::vector<double>& y);
            
            // Prediction
            std::vector<double> predict(const std::vector<double>& x) const;

            // Getters
            double get_intercept() const { return intercept; }
            double get_slope() const { return slope; }
            const std::vector<double>& get_coef() const { return coefficients; }
            const std::vector<double>& get_stats() const { return stats; }
            bool is_fitted() { return is_fitted; }
    };
}