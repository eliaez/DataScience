#pragma once

#include <vector>

class Dataframe;

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
    
    // Log loss for multiple categories with y (one hot encoded) and prob col major 
    double cat_logloss(const std::vector<double>& y, const std::vector<double>& prob, int K);

    // Log likehood for mutliple categories with y (one hot encoded) and prob col major 
    double logLikehood(const std::vector<double>& y, const std::vector<double>& prob, int K);

    // Log likehood null for multiple categories with y (one hot encoded)
    double logLikehood_null(const std::vector<double>& y, int K);

    // Fisher matrix (symmetrical) with x col major (with const) and y_pred from predict_proba col major  
    Dataframe fisher_mat(const Dataframe& x, const Dataframe& y_pred);

    Dataframe cov_mat(Dataframe& fisher);

    // Function to get stderr vector with K category idx
    std::vector<double> stderr_coeff(Dataframe& fisher, int K);
    std::vector<double> stderr_coeff(const Dataframe& cov, int K);

    // Function to get stderr vector with x col major (with const), y_pred from predict_proba col major and K category idx
    std::vector<double> stderr_coeff(const Dataframe& x, const Dataframe& y_pred, int K);
    
    // R2 McFadden
    double mc_fadden();

    // Chi**2
    double chi2();
}