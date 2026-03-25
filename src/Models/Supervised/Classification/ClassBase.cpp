#include <set>
#include <cmath>
#include <stdexcept>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Models/Supervised/Classification/ClassBase.hpp"

using namespace Utils;

namespace Class {

std::string CoeffStats::significance(double p_val) const {
    if (p_val < 0.001) return "***";
    if (p_val < 0.01)  return "** ";
    if (p_val < 0.05)  return "*  ";
    if (p_val < 0.10)  return ".  ";
    return "   ";
}

void ClassificationBase::basic_verif(const Dataframe& x) const {
    if (x.get_rows() == 0 || x.get_cols() == 0 || x.get_cols() < 1) {
        throw std::invalid_argument("Need non-empty input");
    }
}

std::vector<double> ClassificationBase::softmax(const Dataframe& X) const {
    Dataframe W = {X.get_cols(), nb_cats, false, coeffs};
    return softmax(X, W);
}

std::vector<double> ClassificationBase::softmax(const Dataframe& X, const Dataframe& W) const {

    size_t n = X.get_rows();

    // Getting ptrs to elem of each col
    std::vector<std::vector<const double*>> W_j(nb_cats);
    for (size_t j = 0; j < nb_cats; j++) 
        W_j[j] = W.getColumnPtrs(j);

    // Getting ptrs to elem of each row
    std::vector<std::vector<const double*>> X_i(n);
    for (size_t i = 0; i < n; i++)
        X_i[i] = X.getRowPtrs(i);

    // Num
    std::vector<double> scores(n * nb_cats);
    for (size_t i = 0; i < n; i++) {
        
        // To avoid later an overflow
        std::vector<double> z(nb_cats);
        for (size_t j = 0; j < nb_cats; j++)
            z[j] = dot(W_j[j], X_i[i]);

        double max_z = *std::max_element(z.begin(), z.end());
        for (size_t j = 0; j < nb_cats; j++)
            scores[i * nb_cats + j] = std::exp(z[j] - max_z);
    }

    // Denom
    std::vector<double> denom(n, 0.0);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < nb_cats; j++)
            denom[i] += scores[i * nb_cats + j];

    // probas
    std::vector<double> y_v(n * nb_cats);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < nb_cats; j++)
            y_v[j * n + i] = scores[i * nb_cats + j] / denom[i];

    // Y_v col major
    return y_v;
}

void ClassificationBase::nb_categories(const Dataframe& Y) {
    
    // Will detect nb of classes in y
    std::set<int> unique_int;
    for (double x : Y.get_data()) {
        if (x == std::floor(x))
            unique_int.insert(static_cast<int>(x));
    }
    nb_cats = unique_int.size();

    // Last one by default and 0 for binary classification
    ref_class_ = (nb_cats == 2) ? 0 : static_cast<int>(nb_cats) - 1;
}

void ClassificationBase::fit(const Dataframe& x, const Dataframe& y) {

    Dataframe X_const = fit_without_stats(x, y);
    compute_stats(x, X_const, y);
}

std::vector<double> ClassificationBase::predict_proba(const Dataframe& x) const {
    basic_verif(x);
    if (!is_fitted) {
        throw std::runtime_error("Need to have trained your model");
    }
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }
    size_t n = x.get_rows();
    size_t p = x.get_cols();

    // Copy our data 
    std::vector<double> x_v = x.get_data();
    
    // Insert an unit col to get intercept value
    x_v.insert(x_v.begin(), n, 1.0);
    
    Dataframe X = {n, p+1, false, std::move(x_v)};
    return softmax(X);
}

std::vector<double> ClassificationBase::predict(const Dataframe& x) const {

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

std::unique_ptr<ClassificationBase> ClassificationBase::create(const std::vector<double>& /*params*/) {
    throw std::logic_error("GridSearch not supported for this model");
}

void ClassificationBase::clean_params() {
    is_fitted = false;
    coeffs = {};
    gen_stats = {};
    coeff_stats = {};
}

}