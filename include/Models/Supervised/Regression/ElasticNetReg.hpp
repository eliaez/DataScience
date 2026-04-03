#pragma once

#include <vector>
#include <string>
#include "RegBase.hpp"

// ---------------Forward Declaration----------------

class Dataframe;

// ---------------------------------------Reg------------------------------------------

namespace Reg {
    class ElasticRegression : public RegressionBase {
        private:
            double alpha_;
            double l1_ratio_;
        
        public:
            ElasticRegression(double alpha = 0.1, double l1_ratio = 0.5) : 
                alpha_(alpha),  
                l1_ratio_(l1_ratio) {};

            // Training Elastic Net Regression with x col-major
            std::pair<Dataframe, Dataframe> fit_without_stats(const Dataframe& x, const Dataframe& y) override;

            // Calculate Stats after fit function
            void compute_stats(const Dataframe& x, Dataframe& x_c, Dataframe& XtXinv, const Dataframe& y) override;

            // Display stats after training
            void summary(bool detailled = false) const override;

            // Get degree of liberty by using x_c (X centered and scaled)
            double effective_df(Dataframe& X_c) const;

            // Function to create new model
            std::unique_ptr<RegressionBase> create(const std::vector<std::variant<double, std::string>>& params) override;

            // Getter
            double get_alpha() const { return alpha_; }
            double get_l1_ratio() const { return l1_ratio_; }

            // Setter
            void set_alpha(double alpha) { alpha_ = alpha; }
            void set_l1_ratio(double l1_ratio) { l1_ratio_ = l1_ratio; }
    };
}