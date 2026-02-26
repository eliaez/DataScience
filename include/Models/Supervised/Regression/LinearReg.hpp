#pragma once

#include <vector>
#include <string>
#include "RegBase.hpp"

// ---------------Forward Declaration----------------

class Dataframe;

// ---------------------------------------Reg------------------------------------------

namespace Reg {
    class LinearRegression : public RegressionBase { 
        private:
            std::string cov_type_;
            std::vector<int> cluster_ids_;
            std::unique_ptr<Dataframe> Omega_;

        protected:
            // Calculate Stats after fit function
            void compute_stats(const Dataframe& x, const Dataframe& x_const, Dataframe& XtXinv, const Dataframe& y) override;
        
        public:
            // Constructor, cov_type : classical, HC3, HAC and cluster.
            // Cluster_ids for cluster version, Omega for GLS
            LinearRegression(
                std::string cov_type = "classical", 
                std::vector<int> cluster_ids = {},
                std::unique_ptr<Dataframe> Omega = nullptr
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