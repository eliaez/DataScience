#pragma once

#include <vector>
#include "Data/Data.hpp"
#include "Stats/stats.hpp"
#include "Linalg/Linalg.hpp"
#include "Utils/ThreadPool.hpp"

namespace Reg {
    struct CoeffStats {
        std::string name;        
        double beta;             
        double stderr_beta;        
        double t_stat;           
        double p_value;          

        // Stars for significance of p values
        std::string significance() const;
    };

    class LinearRegression {
        private:
            bool is_fitted;
            std::vector<double> coeffs;
            std::vector<double> gen_stats;              // General stats
            std::vector<CoeffStats> coeff_stats;        // Stats for each Coeff
        
            // Function to verify if x non-empty,...
            void basic_verif(const Dataframe& x) const;

            // Predict function with avx2 if enabled, naive if not
            std::vector<double> predict_avx2(const Dataframe& x) const;

            #ifdef __AVX2__
                // Predict function with avx2th
                std::vector<double> predict_avx2th(const Dataframe& x) const;
            #endif

        public:
            // Constructor 
            LinearRegression() : is_fitted(false) {}; // Init to get col major or warn user 

            // Training
            void fit(const Dataframe& x, const Dataframe& y);
            
            // Prediction
            std::vector<double> predict(const Dataframe& x) const;

            // Calculate Stats after fit function
            void compute_stats(const Dataframe& x, const Dataframe& XtXinv, const Dataframe& y);

            // Getters
            double get_intercept() const { return coeffs[0]; }
            const std::vector<double>& get_coeffs() const { return coeffs; }
            const std::vector<double>& get_stats() const { return gen_stats; }
            bool is_model_fitted() { return is_fitted; }
            const std::vector<CoeffStats>& get_coefficient_stats() const { return coeff_stats; }
    };
}