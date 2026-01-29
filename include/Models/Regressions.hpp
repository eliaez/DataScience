#pragma once

#include <vector>
#include "Stats/stats.hpp"
#include "Data/Data.hpp"
#include "Linalg/Linalg.hpp"

namespace Reg {
    class LinearRegression {
        private:
            double intercept;
            double slope;
            std::vector<double> v_stats; // May change
            bool is_fitted;
        
            // Function to verify if x non-empty,...
            void basic_verif(const Dataframe& x) const;

        public:
            // Constructor 
            LinearRegression() : is_fitted(false) {}; // Init to get col major or warn user 

            // Training
            void fit(const Dataframe& x, const Dataframe& y);
            
            // Prediction
            std::vector<double> predict(const Dataframe& x) const;

            // Getters
            double get_intercept() const { return intercept; }
            double get_slope() const { return slope; }
            const std::vector<double>& get_stats() const { return v_stats; }
            bool is_model_fitted() { return is_fitted; }
    };
}