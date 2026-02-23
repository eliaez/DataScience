#pragma once

#include <vector>
#include <string>
#include "RegBase.hpp"

// ---------------Forward Declaration----------------

class Dataframe;

// ---------------------------------------Reg------------------------------------------

namespace Reg {
    class RidgeRegression : public RegressionBase {
        private:
            double lambda_; // L2 Penality

        protected:
            // Calculate Stats after fit function
            void compute_stats(const Dataframe& x, const Dataframe& x_const, Dataframe& XtXinv, const Dataframe& y) override;
        
        public:
            RidgeRegression(double lambda = 1.0) : lambda_(lambda) {};

            // Training Ridge Regression with x col-major
            void fit(const Dataframe& x, const Dataframe& y) override;
            std::pair<Dataframe, Dataframe> fit_without_stats(const Dataframe& x, const Dataframe& y) override;

            // Display stats after training
            void summary(bool detailled = false) const override;
    };
}