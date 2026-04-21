#pragma once

#include <vector>
#include <string>
#include "RegBase.hpp"

// ---------------Forward Declaration----------------

class Dataframe;

// ---------------------------------------Reg------------------------------------------

namespace Reg {
    class StepwiseRegression : public RegressionBase {
        private:
            std::pair<double, double> alpha_;
            std::string method_;
            std::string threshold_;
            std::vector<double> selected_features;

            // Function for Forward
            std::vector<double> forward_reg(const Dataframe& x, const Dataframe& y);

            // Function for Backward
            std::vector<double> backward_reg(const Dataframe& x, const Dataframe& y);

            // Function for Stepwise
            std::vector<double> stepwise_reg(const Dataframe& x, const Dataframe& y);
        
        public:
            // alpha pair represents alpha in, alpha out conditions to keep or erase feature accordingly to
            // the method Forward/Backward/Stepwise and is linked to the threshold variable alpha by default.
            // You can also choose for threshold : "aic"/"bic"
            // "forward" / "backward" (by default) / "stepwise"  
            StepwiseRegression(
                std::pair<double, double> alpha = {0.05, 0.10}, 
                std::string method = "backward", 
                std::string threshold = "alpha") : 
                alpha_(alpha),  
                method_(method),
                threshold_(threshold) {};

            // Training Stepwise Regression with x col-major
            std::pair<Dataframe, Dataframe> fit_without_stats(const Dataframe& x, const Dataframe& y) override;

            // Calculate Stats after fit function
            void compute_stats(const Dataframe& x, Dataframe& x_const, Dataframe& XtXinv, const Dataframe& y) override;
            
            // Function to know nb of selected features
            double effective_df() const;

            // Display stats after training
            void summary(bool detailled = false) const override;

            // Getter
            std::pair<double, double> get_alpha() const { return alpha_; }
            std::string get_method() const { return method_; }
            std::string threshold() const { return threshold_; }

            // Setter
            void set_alpha(std::pair<double, double> alpha) { alpha_ = alpha; }
            void set_method(std::string method) { method_ = method; }
            void get_l1_ratio(std::string threshold) { threshold_ = threshold; }
    };
}