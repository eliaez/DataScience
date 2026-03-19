#include <cmath>
#include <string>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include "Utils/Utils.hpp"
#include "Stats/stats_class.hpp"

using namespace Utils;

namespace Stats_class {

std::vector<double> confusion_matrix(const std::vector<double>& y, const std::vector<double>& y_pred) {
    if (y.empty()) {
        throw std::invalid_argument("Cannot calculate Confusion matrix of empty vector");
    }

    size_t n = y.size();
    if (n != y_pred.size()) {
        throw std::invalid_argument("Y and Ypred need to have the same length");
    }

    // Calc confusion matrix
    double TP = 0.0, FN = 0.0, FP = 0.0, TN = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (y[i] == 1 && y_pred[i] == 1) TP++;
        else if (y[i] == 0 && y_pred[i] == 0) TN++;
        else if (y[i] == 0 && y_pred[i] == 1) FP++;
        else if (y[i] == 1 && y_pred[i] == 0) FN++;
    } 
    return {TP, FN, FP, TN};
}

double accuracy(double TP, double FN, double FP, double TN) {
    return (TP + TN) / (TP + TN + FP + FN);
}

double accuracy(const std::vector<double>& conf_matrix) {
    return accuracy(conf_matrix[0], conf_matrix[1], conf_matrix[2], conf_matrix[3]);
}

double precision(double TP, double FP) {
    return TP / (TP + FP);
}

double recall(double TP, double FN) {
    return TP / (TP + FN);
}

double specificity(double TN, double FP) {
    return TN / (TN + FP);
}

double f1(double precision, double recall) {
    return 2 * precision * recall / (precision + recall);
}

double roc_auc(const std::vector<double>& y, const std::vector<double>& prob) {
    if (y.empty()) {
        throw std::invalid_argument("Cannot calculate Confusion matrix of empty vector");
    }

    size_t n = y.size();
    if (n != prob.size()) {
        throw std::invalid_argument("Y and prob need to have the same length");
    }

    // Calculate roc points
    std::vector<std::pair<double, double>> rocPoints;
    rocPoints.reserve(101);
    for (double threshold = 0.0; threshold <= 1.0; threshold += 0.01) {
        double TP = 0, FP = 0, TN = 0, FN = 0;

        for (size_t i = 0; i < prob.size(); i++) {
            double predicted = (prob[i] >= threshold) ? 1 : 0;

            if (predicted == 1 && y[i] == 1) TP++;
            else if (predicted == 1 && y[i] == 0) FP++;
            else if (predicted == 0 && y[i] == 0) TN++;
            else if (predicted == 0 && y[i] == 1) FN++;
        }

        double tpr = (TP + FN > 0) ? TP / (TP + FN) : 0.0;
        double fpr = (FP + TN > 0) ? FP / (FP + TN) : 0.0;

        rocPoints.push_back({tpr, fpr});
    }

    // Sort by FPR
    std::sort(rocPoints.begin(), rocPoints.end(), 
        [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
            return a.second < b.second;
        }
    );

    // Calc AUC
    double auc = 0.0;
    for (size_t i = 1; i < rocPoints.size(); ++i) {
        double dFPR = rocPoints[i].second - rocPoints[i-1].second;
        double avgTPR = (rocPoints[i].first + rocPoints[i-1].first) / 2.0;
        auc += dFPR * avgTPR;
    }
    return auc;

}

double mcc(double TP, double FN, double FP, double TN) {
    return ((TP * TN) - (FP * FN)) / std::sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
}

double mcc(const std::vector<double>& conf_matrix) {
    return mcc(conf_matrix[0], conf_matrix[1], conf_matrix[2], conf_matrix[3]);
}

double cat_logloss(const std::vector<double>& y, const std::vector<double>& prob, int K) {
    if (K < 2) throw std::invalid_argument("Invalid K:" + std::to_string(K));
    if (y.empty()) {
        throw std::invalid_argument("Cannot calculate Confusion matrix of empty vector");
    }

    size_t n = y.size(); // N * K
    size_t N = n / K;
    if (n != prob.size() || n % K != 0) {
        throw std::invalid_argument("Incompatible sizes");
    }

    double max_y = *std::max_element(y.begin(), y.end());
    if (max_y > 1)
        throw std::invalid_argument("y should be one-hot encoded");

    // Log loss
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (y[i] == 1)
            sum += y[i] * std::log(prob[i] + 1e-15);
    }
    return - sum / N;
}

double logLikehood(const std::vector<double>& y, const std::vector<double>& prob, int K) {
    size_t n = y.size(); // N * K
    size_t N = n / K;
    return - cat_logloss(y, prob, K) * N;
}

}