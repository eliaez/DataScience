#pragma once

#include <vector>
#include <string>
#include "ClassBase.hpp"

// ---------------Forward Declaration----------------

class Dataframe;

// ---------------------------------------Reg------------------------------------------

namespace Class {

    namespace detail {
        struct Node_ {
            double value;
            double threshold;
            bool nan_goes_left;
            size_t feature_index;
            std::unique_ptr<Node_> left;
            std::unique_ptr<Node_> right;
        };

        class DecisionTree_ {
            private:
                double gamma_;               // Reduct min loss for splitting
                double alpha_;               // L1 Regularization
                double lambda_;              // L2 Regularization
                double max_depth_;           // Max depth of a tree
                double min_child_weight_;    // Nb min of weight in each leaf

                size_t total_size;
                std::unique_ptr<Node_> root;
                std::vector<double> features_importance;

                // Create our tree
                std::unique_ptr<Node_> grow(
                    const std::vector<std::vector<const double*>>& X_cols,
                    const std::vector<double>& g,
                    const std::vector<double>& h,
                    double depth        
                );

                // Returns vector of bool, true for left with X being a col
                std::vector<bool> split(const std::vector<const double*> X_col, double threshold) const;

                double leaf_value(const std::vector<double>& g, const std::vector<double>& h) const;

                // Selects the best split according to sample by using IG
                // Returns best_idx, best_threshold and IG
                std::tuple<size_t, double, double, bool> best_split(
                    const std::vector<std::vector<const double*>>& X_cols, 
                    const std::vector<double>& g,
                    const std::vector<double>& h
                ) const;

            public:
                DecisionTree_(
                    double gamma = 0,
                    double alpha = 0, 
                    double lambda = 1,
                    double max_depth = -1,
                    double min_child_weight = 1) 
                : gamma_(gamma), alpha_(alpha), lambda_(lambda), max_depth_(max_depth), min_child_weight_(min_child_weight) {};
                
                // Function to get a leaf value with X being a row
                double traverse(const std::vector<const double*>& X_row, Node_* node) const;
                
                double gain(double GL, double HL, double GR, double HR) const;    

                // Fit to create your Tree with grow fct
                void fit(
                    const std::vector<std::vector<const double*>>& X_cols, 
                    const std::vector<double>& g,
                    const std::vector<double>& h
                );

                // Predicts for each obs with X vector of rows
                std::vector<double> predict(const std::vector<std::vector<const double*>>& X_rows) const;

                // Getter
                const std::vector<double>& get_feature_imp() const { return features_importance; }
        };
    }

    class XGBoost : public ClassificationBase {
        private:
            double gamma_;               // Reduct min loss for splitting
            double alpha_;               // L1 Regularization
            double lambda_;              // L2 Regularization
            double max_depth_;           // Max depth of a tree
            double n_estimators_;        // Nb of tree
            double min_child_weight_;    // Nb min of weight in each leaf

            std::vector<detail::DecisionTree_> forest;
            std::vector<double> softmax(const std::vector<double>& y_pred, size_t n) const;
            std::vector<double> get_logits(const std::vector<std::vector<const double*>>& X_rows) const;

        public:
            XGBoost(
                double gamma = 0,
                double alpha = 0, 
                double lambda = 1,
                double n_estimator = 100,
                double max_depth = -1,
                double min_child_weight = 1) 
            : gamma_(gamma), alpha_(alpha), lambda_(lambda), n_estimators_(n_estimator), 
              max_depth_(max_depth), min_child_weight_(min_child_weight) {};

            // Training XGBoost with x col-major, returns features_imp
            Dataframe fit_without_stats(const Dataframe& x, const Dataframe& y) override;

            // Calculate Stats after fit function
            void compute_stats(const Dataframe& x, Dataframe& features_imp, const Dataframe& y) override;

            // Predict XGBoost
            std::vector<double> predict(const Dataframe& x) const override;
            std::vector<double> predict_proba(const Dataframe& x) const;

            // Display stats after training
            void summary(bool detailled = false) const override;

            // Function to create new model
            std::unique_ptr<ClassificationBase> create(const std::vector<std::variant<double, std::string>>& params) override;
    };
}