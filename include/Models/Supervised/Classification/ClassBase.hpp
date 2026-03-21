#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <functional>

// ---------------Forward Declaration----------------

class Dataframe;


// ---------------------------------------Class------------------------------------------

namespace Class {
    struct CoeffStats {
        std::string category;

        std::vector<std::string> name;        
        std::vector<double> beta;       
        std::vector<double> odds_ratio;      
        std::vector<double> stderr_beta;        
        std::vector<double> z_stat;           
        std::vector<double> p_value;    
        
        std::vector<double> gen_stats; // precision, recall, specificity, f1, roc_auc

        // Stars for significance of p values
        std::string significance(double p_val) const;
    };

    class ClassificationBase {
        protected:
            int nb_cats;
            double tol_;                                   
            int ref_class_;                             // Last one by default
            bool is_fitted;
            double max_iter_;
            double learning_r_;
            std::vector<double> coeffs;
            std::vector<double> gen_stats;              // General stats
            std::vector<CoeffStats> coeff_stats;        // Stats for each Coeff
        
            // Function to verify if x non-empty,...
            void basic_verif(const Dataframe& x) const;

            // Calculate Stats after fit function
            virtual void compute_stats(const Dataframe& x, Dataframe& x_const, const Dataframe& y) = 0;
            
            // Function to handle Softmax in case of multi classes
            std::vector<double> softmax(const Dataframe& X) const;
            std::vector<double> softmax(const Dataframe& X, const Dataframe& W) const;

            // Function to get number of categories
            void nb_categories(const Dataframe& Y);

        public:
            // Constructor 
            ClassificationBase() : is_fitted(false), nb_cats(-1), ref_class_(-1), tol_(1e-4), max_iter_(1000), learning_r_(0.1) {}; // Init to get col major or warn user 
            virtual ~ClassificationBase() = default;

            virtual void fit(const Dataframe& x, const Dataframe& y);
            virtual Dataframe fit_without_stats(const Dataframe& x, const Dataframe& y) = 0;
      
            // Prediction
            virtual std::vector<double> predict(const Dataframe& x) const;
            virtual std::vector<double> predict_proba(const Dataframe& x) const;

            // Display stats after training
            virtual void summary(bool detailled = false) const = 0;

            // Function to create new model 
            virtual std::unique_ptr<ClassificationBase> create(const std::vector<double>& params);

            // Function to clean params from ClassificationBase
            void clean_params();

            // Getters
            double get_tol() const { return tol_; }
            int get_refclass() const { return ref_class_; }
            double get_maxiter() const { return max_iter_; }
            bool is_model_fitted() const { return is_fitted; }
            double get_intercept() const { return coeffs[0]; }
            double get_learningrate() const { return learning_r_; }
            const std::vector<double>& get_coeffs() const { return coeffs; }
            const std::vector<double>& get_stats() const { return gen_stats; }
            const std::vector<CoeffStats>& get_coefficient_stats() const { return coeff_stats; }

            // Setters
            void set_tol(double tol) { tol_ = tol; }
            void set_maxiter(double max_iter) { max_iter_ = max_iter; }
            void set_refclass(int ref_class) { ref_class_ = ref_class; }
            void set_learningrate(double learning_rate) {learning_r_ = learning_rate; }

    };
}