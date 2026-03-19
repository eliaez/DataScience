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
            double penality_;  //  1 (L1) or 2 (L2)
            

        protected:
            // Calculate Stats after fit function
            void compute_stats(const Dataframe& x, Dataframe& x_const, const Dataframe& X_T, const Dataframe& y) override;
        
        public:
            // C = 1/lambda, penality = 1 for L1 or 2 for L2 
            LogisticRegression(double C = 1.0, double penality = 2.0) : C_(C), penality_(penality) {};

            // Training Lasso Regression with x col-major
            std::pair<Dataframe, Dataframe> fit_without_stats(const Dataframe& x, const Dataframe& y) override;

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

            // Setter
            void set_c(double C) { C_ = C; }
            void set_penality(double penality) { penality_ = penality; }
    };
}