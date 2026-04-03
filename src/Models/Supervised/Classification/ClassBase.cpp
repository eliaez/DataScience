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

std::vector<double> ClassificationBase::predict_proba(const Dataframe& /*x*/) const {
    throw std::runtime_error("predict_proba not implemented for this class");
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

std::unique_ptr<ClassificationBase> ClassificationBase::create(const std::vector<std::variant<double, std::string>>& /*params*/) {
    throw std::logic_error("GridSearch not supported for this model");
}

void ClassificationBase::clean_params() {
    is_fitted = false;
    coeffs.clear();
    gen_stats.clear();
    coeff_stats.clear();
}

}