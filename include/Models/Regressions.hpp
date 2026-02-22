#pragma once

#include <vector>
#include <utility>

// ---------------Forward Declaration----------------

class Dataframe;


// ---------------------------------------Reg------------------------------------------

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

    class RegressionBase {
        protected:
            bool is_fitted;
            std::vector<double> coeffs;
            std::vector<double> gen_stats;              // General stats
            std::vector<CoeffStats> coeff_stats;        // Stats for each Coeff
        
            // Function to verify if x non-empty,...
            void basic_verif(const Dataframe& x) const;

            // Calculate Stats after fit function
            virtual void compute_stats(const Dataframe& x, const Dataframe& x_const, Dataframe& XtXinv, const Dataframe& y) = 0;

        public:
            // Constructor 
            RegressionBase() : is_fitted(false) {}; // Init to get col major or warn user 
            virtual ~RegressionBase() = default;

            virtual void fit(const Dataframe& x, const Dataframe& y) = 0;
            virtual std::pair<Dataframe, Dataframe> fit_without_stats(const Dataframe& x, const Dataframe& y) = 0;
      
            // Prediction
            virtual std::vector<double> predict(const Dataframe& x) const;

            // Display stats after training
            virtual void summary(bool detailled = false) const = 0;

            // Getters
            bool is_model_fitted() { return is_fitted; }
            double get_intercept() const { return coeffs[0]; }
            const std::vector<double>& get_coeffs() const { return coeffs; }
            const std::vector<double>& get_stats() const { return gen_stats; }
            const std::vector<CoeffStats>& get_coefficient_stats() const { return coeff_stats; }
    };

    class LinearRegression : public RegressionBase { 
        private:
            std::string cov_type_;
            std::vector<int> cluster_ids_;
            Dataframe Omega_;

        protected:
            // Calculate Stats after fit function
            void compute_stats(const Dataframe& x, const Dataframe& x_const, Dataframe& XtXinv, const Dataframe& y) override;
        
        public:
            // Constructor, cov_type : classical, HC3, HAC and cluster.
            // Cluster_ids for cluster version, Omega for GLS
            LinearRegression(
                std::string cov_type = "classical", 
                std::vector<int> cluster_ids = {},
                Dataframe Omega = {}
            ) : cov_type_(cov_type),
                cluster_ids_(std::move(cluster_ids)),
                Omega_(std::move(Omega)) {};

            // Training OLS / GLS (WLS, FGLS) with x col-major
            void fit(const Dataframe& x, const Dataframe& y) override;
            std::pair<Dataframe, Dataframe> fit_without_stats(const Dataframe& x, const Dataframe& y) override;

            // Display stats after training
            void summary(bool detailled = false) const override;
    };
}