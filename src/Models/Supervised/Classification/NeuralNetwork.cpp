#include <set>
#include <map>
#include <random>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Stats/stats_reg.hpp"
#include "Stats/stats_class.hpp"
#include "Models/Supervised/Classification/NeuralNetwork.hpp"

using namespace Utils;

namespace Class {

std::pair<Dataframe, Dataframe> NeuralNetwork::mini_batches(const Dataframe& X, const Dataframe& y) {
    
    size_t n = X.get_rows();
    size_t p = X.get_cols();

    // Shuffle all idx
    if (batch_idx == 0 || batch_idx >= n) {
        shuffled_idx.resize(n);
        std::iota(shuffled_idx.begin(), shuffled_idx.end(), 0);
        std::shuffle(shuffled_idx.begin(), shuffled_idx.end(), rng);
        batch_idx = 0;
    }

    size_t actual_batch = std::min(static_cast<size_t>(batch_size_), n - batch_idx);
    
    // Creating our dataframes
    std::vector<double> X_v(actual_batch * p);
    std::vector<double> y_v;
    y_v.reserve(actual_batch);

    for (size_t i = 0; i < actual_batch; i++) {

        size_t row_idx = shuffled_idx[batch_idx + i];
        for (size_t j = 0; j < p; j++) {
            X_v[j * actual_batch + i] = X.at(j * n + row_idx);
        }
        y_v.push_back(y.at(row_idx));
    }

    batch_idx += actual_batch;
    Dataframe X_sample = {actual_batch, p, false, std::move(X_v)};
    Dataframe y_sample = {actual_batch, 1, false, std::move(y_v)};
    return {X_sample, y_sample};
}

double NeuralNetwork::compute_cost(const Dataframe& AL, const Dataframe& Y) {

    // If L2 reg
    if (lambda_ != 0.0) {
        double logloss = Stats_class::Mult::logloss_mult(Y.get_data(), AL);

        double L2_term = 0.0;
        for (size_t i = 0; i < layers.size(); i++) {
            L2_term += Lnorm(layers[i].get_W().get_data(), 2, 2);
        }
        return lambda_ * L2_term / (2.0 * static_cast<double>(batch_size_)) + logloss;
    }
    return Stats_class::Mult::logloss_mult(Y.get_data(), AL);
}

Dataframe NeuralNetwork::forward(const Dataframe& X) {

    Dataframe AL = X;
    for (auto& layer : layers) {
        AL = layer.forward(AL); 
    }
    return AL;
}

void NeuralNetwork::backward(const Dataframe& AL, const Dataframe& Y) {

    Dataframe dA = layers.back().backward(AL, Y, true);
    for (int i = (int)layers.size() - 2; i >= 0; i--) {
        dA = layers[i].backward(dA, Y, false);
    }
}

Dataframe NeuralNetwork::fit_without_stats(const Dataframe& x, const Dataframe& y) {
    
    // Tests
    basic_verif(x);
    basic_verif(y);
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }

    nb_categories(y);
    if (y.get_cols() > 1) {
        throw std::invalid_argument("NeuralNetwork doesn't support Y One-Hot");
    }

    size_t n = x.get_rows();

    // Init Layers
    layers.clear();
    for (size_t i = 0; i < activ_.size(); i++) {
        Layer L(
            layer_size_[i],
            layer_size_[i + 1],
            nb_cats,
            activ_[i],
            lambda_,
            beta1_,
            beta2_,
            learning_r_,
            keep_prob_
        );
        L.init_components();
        layers.push_back(L);
    }

    if (n_epochs_ == 0) n_epochs_ = 500;

    // Train
    int t = 1;
    std::vector<double> loss_history;
    loss_history.reserve(n_epochs_);
    for (size_t epoch = 0; epoch < n_epochs_; epoch++) {
        
        batch_idx = 0;
        int n_batches = 0;
        double epoch_loss = 0.0;
        while (batch_idx < n) {
            
            // mini batch
            auto [X_batch, y_batch] = mini_batches(x, y);
            
            // Forward
            Dataframe AL = forward(X_batch);

            // Loss
            double loss = compute_cost(AL, y_batch);
            epoch_loss += loss;
            n_batches++;

            // Backward
            backward(AL, y_batch);

            // Update
            for (auto& layer : layers) layer.update_adam(t);
            t++;
        }

        epoch_loss /= n_batches;
        loss_history.push_back(epoch_loss);

        if (n_epochs_ == 500) {
            if (loss_history.size() > 1 && std::abs(loss_history.back() - loss_history[loss_history.size()-2]) < tol_) break;

            if (epoch % 20 == 0)
                std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss << std::endl;
        }
        else {
            if (epoch % 10 == 0)
                std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss << std::endl;
        }
    }

    // Going through each layer to save W_T for predict function
    for (auto& layer : layers) {
        layer.save_W_T();
    }

    // Results
    is_fitted = true;
    return {loss_history.size(), 1, false, std::move(loss_history)};
}

std::vector<double> NeuralNetwork::predict(const Dataframe& x) const {
    size_t n = x.get_rows();
    std::vector<double> y_pred(n, 0.0);
    std::vector<double> proba = predict_proba(x);

    for (size_t i = 0; i < n; i++) {
        double max_proba = -1.0;
        size_t best_class = 0;

        // Getting best proba of each obs i
        for (size_t k = 0; k < nb_cats; k++) {
            if (proba[k * n + i] > max_proba) {
                max_proba = proba[k * n + i];
                best_class = k;
            }
        }
        y_pred[i] = static_cast<double>(best_class);
    }
    return y_pred;
}

std::vector<double> NeuralNetwork::predict_proba(const Dataframe& x) const {
    basic_verif(x);
    if (!is_fitted) {
        throw std::runtime_error("Need to have trained your model");
    }
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }

    // Predict
    Dataframe A_prev = x;
    for (const auto& layer : layers) {
        A_prev = layer.forward_predict(A_prev);
    }
    return A_prev.get_data();
}

void Layer::relu() {
    size_t tot = Z.get_cols() * Z.get_rows();
    for (size_t i = 0; i < tot; i++) {
        A.at(i) = std::max(0.0, Z.at(i));
    }
}

void NeuralNetwork::compute_stats(const Dataframe& x, Dataframe& loss_history, const Dataframe& y) {
    gen_stats.clear();
    coeff_stats.clear();
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    size_t K = nb_cats == 2 ? 1 : nb_cats;
    
    // Predict 
    std::vector<double> y_proba = predict_proba(x);
    Dataframe Y_proba = {n, nb_cats, false, std::move(y_proba)};

    std::vector<double> y_pred = predict(x);
    Dataframe Y_pred = {n, 1, false, std::move(y_pred)};
    
    // Confusion matrix
    std::vector<double> conf_matrix;
    if (nb_cats == 2) conf_matrix = Stats_class::conf_matrix(y.get_data(), Y_pred.get_data());
    else conf_matrix = Stats_class::Mult::conf_matrix_mult(y.get_data(), Y_pred, K);

    // Roc Auc 
    std::vector<double> roc_auc;
    if (nb_cats == 2) {
        std::vector<double> y_proba_bin(n);
        for (size_t i = 0; i < n; i++)
            y_proba_bin[i] = Y_proba.at(n + i);
        
        roc_auc.push_back(Stats_class::roc_auc(y.get_data(), y_proba_bin));
    }
    else roc_auc = Stats_class::Mult::roc_auc_mult(y.get_data(), Y_proba);

    // If we have not the cols name
    std::vector<std::string> headers(p, "");
    if (x.get_headers().empty()) {
        for (size_t i = 0; i < p; i++) headers[i] = "c" + std::to_string(i);
    }
    else {
        headers = {};
        headers.insert(headers.end(), x.get_headers().begin(), x.get_headers().end());
    }

    if (nb_cats == 2) {
        // Coeff stats
        CoeffStats c;
        for (size_t i = 0; i < p; i++) {
            c.name.push_back(headers[i]);
        }

        double TP = conf_matrix[0];
        double FN = conf_matrix[1];
        double FP = conf_matrix[2];
        double TN = conf_matrix[3];

        double prec = Stats_class::precision(TP, FP);
        double rec  = Stats_class::recall(TP, FN);
        double spec = Stats_class::specificity(TN, FP);
        double f1_  = Stats_class::f1(prec, rec);

        c.gen_stats = {prec, rec, spec, f1_, roc_auc[0]};
        coeff_stats.push_back(c);

        double logL = Stats_class::logLikelihood(y.get_data(), Y_proba);
        double logloss = - logL / n;
        double accuracy = (TP + TN) / n;

        gen_stats.push_back(logL);
        gen_stats.push_back(logloss);
        gen_stats.push_back(loss_history.get_data().back());
        gen_stats.push_back(Stats_class::mcc(conf_matrix));
        gen_stats.push_back(accuracy);
        gen_stats.push_back(prec);
        gen_stats.push_back(rec);
        gen_stats.push_back(spec);
        gen_stats.push_back(f1_);
        gen_stats.push_back(roc_auc[0]);
    }
    else {
        // For each category
        std::vector<double> f1;
        std::vector<double> recall;
        std::vector<double> precision;
        std::vector<double> specificity;
        f1.reserve(K);
        recall.reserve(K);
        precision.reserve(K);
        specificity.reserve(K);
        for (auto cat : rangeExcept(K, K)) {
            
            // Save our stats
            CoeffStats c;
            c.category = "Class " + std::to_string(cat);
            
            for (size_t i = 0; i < p; i++) {
                c.name.push_back(headers[i]);
            }

            double TP = conf_matrix[cat * K + cat];
            double FP = 0, FN = 0, TN = 0;
            for (size_t j = 0; j < K; j++) {
                if (j != cat) {
                    FP += conf_matrix[j * K + cat];  
                    FN += conf_matrix[cat * K + j];
                }
            }
            TN = n - TP - FP - FN;
            precision.push_back(Stats_class::precision(TP, FP));
            recall.push_back(Stats_class::recall(TP, FN));
            specificity.push_back(Stats_class::specificity(TN, FP));
            f1.push_back(Stats_class::f1(precision.back(), recall.back()));

            c.gen_stats.push_back(precision.back());
            c.gen_stats.push_back(recall.back());
            c.gen_stats.push_back(specificity.back());
            c.gen_stats.push_back(f1.back());
            c.gen_stats.push_back(roc_auc[cat]);
            coeff_stats.push_back(c);
        }
        double logL = Stats_class::logLikelihood(y.get_data(), Y_proba);
        double logloss = - logL / n;

        // Accuracy
        double count = 0.0;
        for (size_t i = 0; i < K; i++) count += conf_matrix[i * K + i];
        double accuracy = count / n;

        // Save general stats
        gen_stats.push_back(logL);
        gen_stats.push_back(logloss);
        gen_stats.push_back(loss_history.get_data().back());
        gen_stats.push_back(Stats_class::Mult::mcc_mult(conf_matrix, n, K));
        gen_stats.push_back(accuracy);
        gen_stats.push_back(mean(precision));
        gen_stats.push_back(mean(recall));
        gen_stats.push_back(mean(specificity));
        gen_stats.push_back(mean(f1));
        gen_stats.push_back(mean(roc_auc));
    }
}

void NeuralNetwork::summary(bool detailled) const {

    std::cout << "\n=== Classification SUMMARY ===\n\n";

    // Global Stats
    std::cout << "Log-Likelyhood Predict = " << gen_stats[0] << "\n";
    std::cout << "Log-Loss Predict       = " << gen_stats[1] << "\n";
    std::cout << "Log-Loss Train         = " << gen_stats[2] << "\n";
    std::cout << "MCC                    = " << gen_stats[3] << "\n";
    std::cout << "Accuracy               = " << gen_stats[4] << "\n";
    std::cout << "Precision              = " << gen_stats[5] << "\n";
    std::cout << "Recall                 = " << gen_stats[6] << "\n";
    std::cout << "Specificity            = " << gen_stats[7] << "\n";
    std::cout << "F1                     = " << gen_stats[8] << "\n";
    std::cout << "ROC AUC                = " << gen_stats[9] << "\n\n";

    if (detailled) {
        for (const auto& stat_ : coeff_stats) {
            std::cout << "--- " << stat_.category << " ---\n";

            // Metrics
            std::cout << "Precision = " << std::fixed << std::setprecision(4) << stat_.gen_stats[0]
                    << "  Recall = "    << stat_.gen_stats[1]
                    << "  Specificity = " << stat_.gen_stats[2]
                    << "  F1 = "        << stat_.gen_stats[3]
                    << "  ROC AUC = "   << stat_.gen_stats[4] << "\n\n";
            std::cout << "\n";
        }
    }
    std::cout << "\n" << std::endl;
}

Dataframe Layer::relu(const Dataframe& Z_) const {

    size_t tot = Z_.get_cols() * Z_.get_rows();
    std::vector<double> res(tot, 0.0);
    for (size_t i = 0; i < tot; i++) {
        res[i] = std::max(0.0, Z_.at(i));
    }
    return {Z_.get_rows(), Z_.get_cols(), false, std::move(res)};
}

void Layer::softmax() {
    size_t n_ = Z.get_rows();
    for (size_t i = 0; i < n_; i++) {

        // Max for numerical stability
        double max_z = -std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < nb_cats_; k++)
            max_z = std::max(max_z, Z.at(k * n_ + i));

        // Exp + sum
        double denom = 0.0;
        for (size_t k = 0; k < nb_cats_; k++) {
            A.at(k * n_ + i) = std::exp(Z.at(k * n_ + i) - max_z);
            denom += A.at(k * n_ + i);
        }

        // Normalize
        for (size_t k = 0; k < nb_cats_; k++)
            A.at(k * n_ + i) /= denom;
    }
}

Dataframe Layer::softmax(const Dataframe& Z_) const {
    size_t n_ = Z_.get_rows();
    std::vector<double> res(n_ * Z_.get_cols(), 0.0);
    for (size_t i = 0; i < n_; i++) {

        // Max for numerical stability
        double max_z = -std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < nb_cats_; k++)
            max_z = std::max(max_z, Z_.at(k * n_ + i));

        // Exp + sum
        double denom = 0.0;
        for (size_t k = 0; k < nb_cats_; k++) {
            res[k * n_ + i] = std::exp(Z_.at(k * n_ + i) - max_z);
            denom += res[k * n_ + i];
        }

        // Normalize
        for (size_t k = 0; k < nb_cats_; k++)
            res[k * n_ + i] /= denom;
    }
    return {Z_.get_rows(), Z_.get_cols(), false, std::move(res)};
}

void Layer::tanh_forward() {
    size_t tot = Z.get_cols() * Z.get_rows();
    for (size_t i = 0; i < tot; i++) {
        A.at(i) = std::tanh(Z.at(i));
    }
}

Dataframe Layer::tanh_forward(const Dataframe& Z_) const {

    size_t tot = Z_.get_cols() * Z_.get_rows();
    std::vector<double> res(tot, 0.0);
    for (size_t i = 0; i < tot; i++) {
        res[i] = std::tanh(Z_.at(i));
    }
    return {Z_.get_rows(), Z_.get_cols(), false, std::move(res)};
}

Dataframe Layer::relu_backward(const Dataframe& dA) const {

    size_t tot = Z.get_cols() * Z.get_rows();
    std::vector<double> dZ(tot, 0.0);
    for (size_t i = 0; i < tot; i++) {
        dZ[i] = Z.at(i) > 0 ? dA.at(i) : 0.0; 
    }
    return {Z.get_rows(), Z.get_cols(), false, std::move(dZ)};
}

Dataframe Layer::tanh_backward(const Dataframe& dA) const {

    size_t tot = Z.get_cols() * Z.get_rows();
    std::vector<double> dZ(tot, 0.0);
    for (size_t i = 0; i < tot; i++) {
        dZ[i] = dA.at(i) * (1 - A.at(i) * A.at(i)); 
    }
    return {Z.get_rows(), Z.get_cols(), false, std::move(dZ)};
}

void Layer::init_components() {
    
    double sigma;
    std::random_device rd;
    std::mt19937 gen(rd());

    // He else Xavier
    if (activ_ == ActivationType::RELU) sigma = std::sqrt(2.0 / n_in_);
    else sigma = std::sqrt(1.0 / n_in_);

    // Random W
    std::vector<double> W_v(n_in_ * n_out_, 0.0);
    std::normal_distribution<double> dist(0.0, sigma);
    for (int i = 0; i < n_in_ * n_out_; i++) W_v[i] = dist(gen);
    W = {static_cast<size_t>(n_out_), static_cast<size_t>(n_in_), true, std::move(W_v)};

    // Init b
    std::vector<double> b_v(n_out_, 0.0);
    b = {1, static_cast<size_t>(n_out_), false, std::move(b_v)};

    // Init Adam
    mW.clear();
    mb.clear();
    mW.assign(n_in_ * n_out_, 0.0);
    mb.assign(n_out_, 0.0);
    
    vW.clear();
    vb.clear();
    vW.assign(n_in_ * n_out_, 0.0);
    vb.assign(n_out_, 0.0);
}

void Layer::update_adam(int t) {

    // Calculate W moments then update it
    size_t tot = dW.get_rows() * dW.get_cols();
    for (size_t i = 0; i < tot; i++) {
        
        // Moments
        mW[i] = beta1_ * mW[i] + (1 - beta1_) * dW.at(i);
        vW[i] = beta2_ * vW[i] + (1 - beta2_) * dW.at(i) * dW.at(i);

        // Bias correction + update W
        W.at(i) = W.at(i) - lr_ * (mW[i] / (1 - std::pow(beta1_, t))) / (std::sqrt(vW[i] / (1 - std::pow(beta2_, t))) + 1e-8);

        // Same for b
        if (i < n_out_) {
            mb[i] = beta1_ * mb[i] + (1 - beta1_) * db.at(i);
            vb[i] = beta2_ * vb[i] + (1 - beta2_) * db.at(i) * db.at(i);
            b.at(i) = b.at(i) - lr_ * (mb[i] / (1 - std::pow(beta1_, t))) / (std::sqrt(vb[i] / (1 - std::pow(beta2_, t))) + 1e-8);
        }
    }
}

void Layer::activ_forward() {
    switch (activ_)
    {
        case ActivationType::RELU:
            relu();
            break;
        case ActivationType::SOFTMAX:
            softmax();
            break;
        case ActivationType::TANH:
            tanh_forward();
            break;
        case ActivationType::LINEAR:
            A = Z;
            break;
        default:
            throw std::invalid_argument("Issue with ActivationType");
            break;
    }
}

void Layer::linear_forward(const Dataframe& A_prev) {
    // A_prev_row
    A_prev_ = A_prev.change_layout();

    // Unit matrix to broadcast b
    std::vector<double> unit(A_prev.get_rows(), 1.0);
    Dataframe unit_mat = {A_prev.get_rows(), 1, true, std::move(unit)};
    
    // Z
    Z = (A_prev_ * (~W)) + (unit_mat * b);

    // Init A
    std::vector<double> A_init(Z.get_rows() * Z.get_cols(), 0.0);
    A = {Z.get_rows(), Z.get_cols(), false, std::move(A_init)};
    
}

void Layer::dropout_forward() {

    M.clear();
    size_t tot = A.get_rows() * A.get_cols();
    M.assign(tot, 0.0);
    std::vector<double> A_drop_v = A.get_data();
    std::bernoulli_distribution dist(keep_prob_);

    // Create our mask and create our A_drop
    for (size_t i = 0; i < tot; i++) {
        M[i] = dist(rng) ? 1.0 / keep_prob_ : 0.0;
        A_drop_v[i] *= M[i];
    }
    A = {A.get_rows(), A.get_cols(), false, std::move(A_drop_v)};
}

Dataframe Layer::forward(const Dataframe& A_prev) {
    linear_forward(A_prev);
    activ_forward();
    if (keep_prob_ != 1.0) dropout_forward();
    return A;
}

Dataframe Layer::forward_predict(const Dataframe& A_prev) const {
    
    // A_prev_row
    Dataframe A_prev_row = A_prev.change_layout();

    // Unit matrix to broadcast b
    std::vector<double> unit(A_prev.get_rows(), 1.0);
    Dataframe unit_mat = {A_prev.get_rows(), 1, true, std::move(unit)};

    // Z
    Dataframe Z_ = (A_prev_row * W_T) + (unit_mat * b);
    Dataframe A_;
    switch (activ_)
    {
        case ActivationType::RELU:
            A_ = relu(Z_);
            break;
        case ActivationType::SOFTMAX:
            A_ = softmax(Z_);
            break;
        case ActivationType::TANH:
            A_ = tanh_forward(Z_);
            break;
        case ActivationType::LINEAR:
            A_ = Z_;
            break;
        default:
            throw std::invalid_argument("Issue with ActivationType");
            break;
    }
    return A_;
}

Dataframe Layer::linear_backward(const Dataframe& dZ) {

    size_t n_ = dZ.get_rows();

    // Setup our inputs
    if (A_prev_.get_storage()) A_prev_.change_layout_inplace();
    std::vector<double> dZ_row_v = mult(dZ.get_data(), 1.0 / n_);
    Dataframe dZ_T = ~Dataframe(Z.get_rows(), dZ.get_cols(), false, std::move(dZ_row_v));

    // dW
    dW = dZ_T.change_layout() * A_prev_;

    // For L2 reg
    if (lambda_ != 0.0) {
        std::vector<double> reg_term = mult(W.get_data(), lambda_ / n_);
        dW = dW + Dataframe(n_out_, n_in_, false, std::move(reg_term));
    }

    // db
    std::vector<double> db_v(n_out_, 0.0);
    for (size_t i = 0; i < n_out_; i++) {
        for (size_t j = 0; j < n_; j++) db_v[i] += dZ.at(j * n_out_ + i) / n_; 
    }
    db = {static_cast<size_t>(n_out_), 1, false, std::move(db_v)};

    // dA_prev
    Dataframe dA_prev = dZ.change_layout() * W;
    return dA_prev;
}

Dataframe Layer::activ_backward(const Dataframe& dA) const {
    
    Dataframe dZ;
    switch (activ_){
        case ActivationType::RELU:
            dZ = relu_backward(dA);
            break;
        case ActivationType::TANH:
            dZ = tanh_backward(dA);
            break;
        case ActivationType::LINEAR:
            dZ = dA;
            break;
        default:
            throw std::invalid_argument("Softmax is only for final layer");
            break;
    }
    return dZ;
}

Dataframe Layer::dropout_backward(const Dataframe& dA) const {

    size_t tot = dA.get_rows() * dA.get_cols();
    std::vector<double> dA_drop_v = dA.get_data();

    // Create our dA_drop
    for (size_t i = 0; i < tot; i++) dA_drop_v[i] *= M[i];
    Dataframe dA_drop = {dA.get_rows(), dA.get_cols(), false, std::move(dA_drop_v)};
    return dA_drop;
}

Dataframe Layer::backward(const Dataframe& dA, const Dataframe& Y, bool is_last) {
    Dataframe dA_drop = dA;
    if (keep_prob_ != 1.0) dA_drop = dropout_backward(dA);

    Dataframe dZ;
    if (is_last) {
        Dataframe Y_onehot = Y;
        Y_onehot.OneHot(0);
        dZ = A - Y_onehot;
    }
    else dZ = activ_backward(dA_drop);
    return linear_backward(dZ);
}
}