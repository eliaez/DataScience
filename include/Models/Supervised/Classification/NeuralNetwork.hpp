#pragma once

#include <vector>
#include <random>
#include <string>
#include "Data/Data.hpp"
#include "ClassBase.hpp"

namespace Class {

    enum ActivationType { RELU, SOFTMAX, TANH, LINEAR };

    class Layer {
        private:
            double beta1_;              // Adam
            double beta2_;              // Adam
            double lambda_;             // L2
            size_t nb_cats_;
            double lr_;
            double keep_prob_;          // Dropout
            int n_in_, n_out_;
            ActivationType activ_;
            Dataframe W, b;             // Inputs
            Dataframe W_T;
            Dataframe dW, db;           // Gradient
            std::vector<double> mW, mb; // Momemtum
            std::vector<double> vW, vb; // Adam 
            Dataframe Z, A, A_prev_;    // Forward (Cache)
            std::vector<double> M;      // Dropout mask 
            std::mt19937 rng{std::random_device{}()};

            // Activation functions
            void relu();
            void softmax();
            void tanh_forward();
            Dataframe relu_backward(const Dataframe& dA) const;
            Dataframe tanh_backward(const Dataframe& dA) const;

            // Forward functions
            void linear_forward(const Dataframe& A_prev);
            void activ_forward();
            void dropout_forward();
            
            // Backward functions
            Dataframe linear_backward(const Dataframe& dZ);
            Dataframe activ_backward(const Dataframe& dA) const;
            Dataframe dropout_backward(const Dataframe& dA) const;

            // To predict
            Dataframe relu(const Dataframe& Z_) const;
            Dataframe softmax(const Dataframe& Z_) const;
            Dataframe tanh_forward(const Dataframe& Z_) const;

        public:
            Layer(
                int n_in, 
                int n_out, 
                size_t nb_cats,
                ActivationType activ,
                double lambda = 0.0,
                double beta1 = 0.9,
                double beta2 = 0.999,
                double lr = 0.1,
                double keep_prob = 1.0) : 
                n_in_(n_in), n_out_(n_out), nb_cats_(nb_cats), activ_(activ), 
                lambda_(lambda), beta1_(beta1), beta2_(beta2), lr_(lr), 
                keep_prob_(keep_prob) {};

            void init_components();
            void update_adam(int t);
            Dataframe forward(const Dataframe& A_prev);
            Dataframe forward_predict(const Dataframe& A_prev) const;
            Dataframe backward(const Dataframe& dA, const Dataframe& Y, bool is_last);
            
            // Function to save W_T for predict function
            void save_W_T() { W_T = ~W; }

            // Getter
            const Dataframe& get_W() const { return W; }
    };
    
    class NeuralNetwork : public ClassificationBase {
        private:
            double beta1_;                      // Adam
            double beta2_;                      // Adam
            double lambda_;                     // L2
            double keep_prob_;                  // Dropout 
            size_t n_epochs_;

            // Layers variables
            std::vector<Layer> layers;
            std::vector<int> layer_size_;
            std::vector<ActivationType> activ_; // Your layer activations type

            // Mini batches variables
            int batch_size_;
            size_t batch_idx;
            std::vector<size_t> shuffled_idx;
            std::mt19937 rng{std::random_device{}()};

            // Function to Forward X to all layers and then return AL 
            Dataframe forward(const Dataframe& X);

            // Function to propagate gradients backward through all layers
            void backward(const Dataframe& AL, const Dataframe& Y);

            // Compute cost for binary/multiple classification and with/without L2 reg
            double compute_cost(const Dataframe& AL, const Dataframe& Y);

            std::pair<Dataframe, Dataframe> mini_batches(const Dataframe& X, const Dataframe& y);
        
        public:
            
            NeuralNetwork(
                std::vector<int> layer_size,
                std::vector<ActivationType> activ,
                size_t n_epochs = 0,
                double lambda = 0.0,
                double beta1 = 0.9,
                double beta2 = 0.999,
                double keep_prob = 1.0,
                int batch_size = 32) : 
                layer_size_(std::move(layer_size)), activ_(std::move(activ)), 
                n_epochs_(n_epochs), lambda_(lambda), beta1_(beta1), beta2_(beta2),
                keep_prob_(keep_prob), batch_size_(batch_size) {};

            // Training NeuralNetwork with x col-major, returns features_imp
            Dataframe fit_without_stats(const Dataframe& x, const Dataframe& y) override;

            // Calculate Stats after fit function
            void compute_stats(const Dataframe& x, Dataframe& loss_history, const Dataframe& y) override;

            // Predict NeuralNetwork
            std::vector<double> predict(const Dataframe& x) const override;
            std::vector<double> predict_proba(const Dataframe& x) const override;

            // Display stats after training
            void summary(bool detailled = false) const override;
    };
}