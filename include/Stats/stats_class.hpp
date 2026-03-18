#pragma once

#include <vector>

namespace Stats_class {

    // Will return TP, FN, FP, TN
    std::vector<double> confusion_matrix(const std::vector<double>& y, const std::vector<double>& y_pred);

    double accuracy(const std::vector<double>& conf_matrix);
    double accuracy(double TP, double FN, double FP, double TN);
    
    double precision(double TP, double FP);

    double recall(double TP, double FN);

    double specificity(double TN, double FP);

    double f1(double precision, double recall);

    double roc_auc(const std::vector<double>& y, const std::vector<double>& prob);

    // Matthews Correlation Coefficient
    double mcc(const std::vector<double>& conf_matrix);
    double mcc(double TP, double FN, double FP, double TN);

    double logloss(const std::vector<double>& y, const std::vector<double>& prob);
}