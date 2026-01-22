#pragma once

#include <vector>

/* Regression Linéaire utilise la méthode des moindres carrés ordinaire et est simplement 
le fait de minimiser la somme des carrés des écarts entre les valeurs prédites et les 
valeurs observées par rapport à nos deux paramètres qui sont l'origine et 
la pente (Cov(X, Y)/Var(X)) dans le cas d'une régression simple */

namespace Reg {
    class LinearRegression {
        private:
            double intercept;
            double slope;
            std::vector<double> coefficients;
            std::vector<double> stats; // May change
            bool is_fitted;

        public:
            // Constructor 
            LinearRegression() : is_fitted(false) {};    

            // Training
            void fit(const std::vector<double>& x, const std::vector<double>& y);
            
            // Prediction
            std::vector<double> predict(const std::vector<double>& x) const;

            // Getters
            double get_intercept() const { return intercept; }
            double get_slope() const { return slope; }
            const std::vector<double>& get_coef() const { return coefficients; }
            const std::vector<double>& get_stats() const { return stats; }
            bool is_fitted() { return is_fitted; }
    };
}