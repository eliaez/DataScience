#pragma once

#include <vector>
#include <string>
#include "ClassBase.hpp"

// ---------------Forward Declaration----------------

class Dataframe;

// ---------------------------------------Reg------------------------------------------

namespace Class {
    class LogisticRegression : public ClassificationBase {
        private:
            double C_;         // 1 / lambda
            double penality_;  // 0 (no penality), 1 (L1), 1.5 (Elastic Net) or 2 (L2)
            double l1_ratio_;  // Used only if elastic net
            

        protected:
            // Calculate Stats after fit function
            void compute_stats(const Dataframe& x, Dataframe& x_const, const Dataframe& y) override;
        
        public:
            // C = 1/lambda, penality = 0 (no penality), 1 (L1), 1.5 (Elastic Net) or 2 (L2)
            // l1_ratio used only if elastic net
            LogisticRegression(double C = 1.0, double penality = 2.0, double l1_ratio = 0.5) 
                : C_(C), penality_(penality), l1_ratio_(l1_ratio) {};

            // Training Lasso Regression with x col-major
            Dataframe fit_without_stats(const Dataframe& x, const Dataframe& y) override;

            // Display stats after training
            void summary(bool detailled = false) const override;

            // Generate a vector of potential C to try with log-scale then 
            // it'll try finding the optimal C through Validation::GridSearchCV
            // Start & end for the range and nb for the steps
            void optimal_c(double start, double end, int nb, const Dataframe& x, const Dataframe& y);

            // Function to create new model
            std::unique_ptr<ClassificationBase> create(const std::vector<double>& params) override;

            // Getter
            double get_c() const { return C_; }
            double get_penality() const { return penality_; }
            double get_l1_ratio() const { return l1_ratio_; }

            // Setter
            void set_c(double C) { C_ = C; }
            void set_penality(double penality) { penality_ = penality; }
            void set_l1_ratio(double l1_ratio) { l1_ratio_ = l1_ratio; }
    };
}