#pragma once

#include <vector>
#include <string>
#include "ClassBase.hpp"

// ---------------Forward Declaration----------------

class Dataframe;

// ---------------------------------------Reg------------------------------------------

namespace Class {

    namespace detail {
        struct Node {
            double value;
            double threshold;
            size_t feature_index;
            std::unique_ptr<Node> left;
            std::unique_ptr<Node> right;
        };

        class DecisionTree {
            private:
                double max_depth_;             // Max depth of a tree
                double max_features_;          // Nb of features to consider for each split
                double min_samples_leaf_;      // Nb min of samples in each leaf
                double min_samples_split_;     // Nb min of samples to split node
                std::string criterion_;     // "gini" or "entropy"

                std::unique_ptr<Node> root;
                std::vector<double> features_importance;

                // Create our tree
                std::unique_ptr<Node> grow(
                    const std::vector<std::vector<const double*>>& X_cols,
                    const std::vector<double>& y,
                    double depth        
                );

                // Returns vector of bool, true for left with X being a col
                std::vector<bool> split(const std::vector<const double*> X_col, double threshold) const;

                // Selects the best split according to sample by using IG
                // Returns best_idx, best_threshold and IG
                std::tuple<size_t, double, double> best_split(const std::vector<std::vector<const double*>>& X_cols, const std::vector<double>& y) const;

            public:
                DecisionTree(
                    double max_features,
                    double max_depth = -1,
                    double min_samples_split = 2,
                    double min_samples_leaf = 1,
                    const std::string& criterion = "entropy") 
                : max_features_(max_features), max_depth_(max_depth),  
                  min_samples_split_(min_samples_split), min_samples_leaf_(min_samples_leaf), 
                  criterion_(criterion) {};

                double gini(const std::vector<double>& y) const;
                double entropy(const std::vector<double>& y) const;
                double information_gain(const std::vector<double>& y, const std::vector<double>& left_y, const std::vector<double>& right_y) const;

                // Majority vote
                static double leaf_value(const std::vector<double>& y);
                
                // Function to get a leaf value with X being a row
                double traverse(const std::vector<const double*>& X_row, Node* node) const;
                
                // Fit to create your Tree with grow fct
                void fit(const std::vector<std::vector<const double*>>& X_cols, const std::vector<double>& y);

                // Predicts for each obs with X vector of rows
                std::vector<double> predict(const std::vector<std::vector<const double*>>& X_rows) const;

                // Getter
                const std::vector<double>& get_feature_imp() const { return features_importance; }
        };
    }

    class RandomForest : public ClassificationBase {
        private:
            double max_depth_;             // Max depth of a tree
            double n_estimators_;          // Nb of tree
            double min_samples_leaf_;      // Nb min of samples in each leaf
            double min_samples_split_;     // Nb min of samples to split node
            std::string criterion_;        // "gini" or "entropy"
            std::string max_features_;     // Nb of features to consider for each split "sqrt", "log2", "all", "0.5"%,...

            std::vector<detail::DecisionTree> forest; 
            std::vector<std::vector<size_t>> oob_indices;

            size_t max_features(size_t p);
            std::vector<size_t> bootstrap(int n) const;
            double majority_vote(const std::vector<double>& y) const;

        protected:
            // Calculate Stats after fit function
            void compute_stats(const Dataframe& x, Dataframe& features_imp, const Dataframe& y) override;
        
        public:
            
            RandomForest(
                double n_estimator = 100,
                double max_depth = -1,
                double min_samples_split = 2,
                double min_samples_leaf = 1,
                const std::string& max_features = "sqrt",
                const std::string& criterion = "gini") 
            : n_estimators_(n_estimator), max_depth_(max_depth), min_samples_split_(min_samples_split), 
              min_samples_leaf_(min_samples_leaf), max_features_(max_features), criterion_(criterion) {};

            // Training RandomForest with x col-major
            Dataframe fit_without_stats(const Dataframe& x, const Dataframe& y) override;

            // Predict RandomForest
            std::vector<double> predict(const Dataframe& x) const override;

            // Display stats after training
            void summary(bool detailled = false) const override;

            // Function to create new model
            std::unique_ptr<ClassificationBase> create(const std::vector<std::variant<double, std::string>>& params) override;
    };
}