#pragma once

#include <vector>
#include <string>
#include "RegBase.hpp"

// ---------------Forward Declaration----------------

class Dataframe;

// ---------------------------------------Reg------------------------------------------

namespace Reg {
    class LassoRegression : public RegressionBase {
        private:
            double lambda_; // L1 Penality

        protected:
            // Calculate Stats after fit function
            void compute_stats(const Dataframe& x, Dataframe& x_c, Dataframe& XtXinv, const Dataframe& y) override;
        
        public:
            LassoRegression(double lambda = 1.0) : lambda_(lambda) {};

            // Training Lasso Regression with x col-major
            void fit(const Dataframe& x, const Dataframe& y) override;
            std::pair<Dataframe, Dataframe> fit_without_stats(const Dataframe& x, const Dataframe& y) override;

            // Display stats after training
            void summary(bool detailled = false) const override;

            // Generate a vector of potential lambdas to try with log-scale then 
            // it'll try finding the optimal lambda through Validation::GridSearchCV
            // Start & end for the range and nb for the steps
            void optimal_lambda(double start, double end, int nb, const Dataframe& x, const Dataframe& y);

            // Function to get non-null coefs following the fitting of lasso 
            int nonzero_coeffs() const;

            // Function to create new model
            std::unique_ptr<RegressionBase> create(const std::vector<double>& params) override;

            // Getter
            double get_lambda() const { return lambda_; }

            // Setter
            void set_lambda(double lambda) { lambda_ = lambda; }
    };
}