#pragma once

#include <vector>
#include <string>
#include "ClassBase.hpp"

// ---------------Forward Declaration----------------

class Dataframe;

// ---------------------------------------Reg------------------------------------------

namespace Class {
    class SVM_Algo : public ClassificationBase {
        private:
            double C_;             // 1 / lambda
            double degree_;        // if poly kernel
            std::string gamma_;    // Point influence "scale" or "auto"
            std::string kernel_;   // "linear", "poly" or "rbf"

            double gamma_val;
            std::vector<double> sv_x;
            std::vector<bool> sv_bool;
            std::vector<double> alpha_;
            std::vector<double> sv_alpha_y;

            Dataframe kernel_meth(const Dataframe& X1, const Dataframe& X2) const;
        
        public:
            // C = 1/lambda, kernel = "linear", "poly" or "rbf", gamma = "scale" or "auto"
            SVM_Algo(double C = 1.0, const std::string& kernel = "linear", const std::string& gamma = "scale", double degree = 2) 
                : C_(C), kernel_(kernel), gamma_(gamma), degree_(degree) {};

            // Training SVM with x col-major
            Dataframe fit_without_stats(const Dataframe& x, const Dataframe& y) override;

            // Calculate Stats after fit function
            void compute_stats(const Dataframe& x, Dataframe& x_const, const Dataframe& y) override;

            // Predict SVM
            std::vector<double> predict(const Dataframe& x) const override;

            // Display stats after training
            void summary(bool detailled = false) const override;

            // Function to create new model
            std::unique_ptr<ClassificationBase> create(const std::vector<std::variant<double, std::string>>& params) override;

            // Getter
            double get_c() const { return C_; }
            double get_degree() const { return degree_; }
            std::string get_gamma() const { return gamma_; }
            std::string get_kernel() const { return kernel_; }
            const std::vector<double>& get_alpha_vect() const { return alpha_; }
            const std::vector<bool>& get_which_is_SV() const { return sv_bool; }

            // Setter
            void set_c(double C) { C_ = C; }
            void set_degree(double degree) { degree_ = degree; }
            void set_gamma(const std::string& gamma) { gamma_ = gamma; }
            void set_kernel(const std::string& kernel) { kernel_ = kernel; }
    };
}