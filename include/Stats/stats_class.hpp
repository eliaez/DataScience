#pragma once

#include <vector>

class Dataframe;

namespace Stats_class {

    // Confusion matrix (K = 2), will return {TP, FN, FP, TN}
    std::vector<double> conf_matrix(const std::vector<double>& y, const std::vector<double>& y_pred);

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

    double normal_pval(double z);

    // Log Likelihood for multiple categories with prob col major 
    double logLikelihood(const std::vector<double>& y, const Dataframe& prob);

    // Log Likelihood null for multiple categories
    double logLikelihood_null(const std::vector<double>& y, int K);

    // Fisher matrix (symmetrical) with x_const col major (with const) and y_proba from predict_proba col major  
    Dataframe fisher_mat(const Dataframe& x_const, const Dataframe& y_proba, size_t ref_class);

    Dataframe cov_mat(Dataframe& fisher);

    // Function to get stderr vector with K for idx of category, p nb of features (with intercept)
    std::vector<double> stderr_coeff(const Dataframe& cov, int K, int p);

    // Function to get stderr vector with x col major (with const), y_pred from predict_proba col major with K for idx of category
    std::vector<double> stderr_coeff(const Dataframe& x, const Dataframe& y_pred, int K);
    
    // R2 McFadden
    double mc_fadden(double loglikelihood_model, double loglikelihood_null);

    // Chi**2 p value with df = (K-1) * p without intercept
    double chi2_pval(double loglikelihood_model, double loglikelihood_null, double df);

    namespace Mult {
        // Confusion matrix multi categories (row major), will vector K*K, row major with conf_mat[i*K + j] = "predict j, true i"
        std::vector<double> conf_matrix_mult(const std::vector<double>& y, const Dataframe& y_pred);
    
        // Log loss for multiple categories with prob col major 
        double logloss_mult(const std::vector<double>& y, const Dataframe& prob);

        // Roc Auc multi categories with prob col major
        std::vector<double> roc_auc_mult(const std::vector<double>& y, const Dataframe& prob);
        
        // Mcc multi categories with conf_matrix row major (from conf_matrix_mult) and K number of categories
        double mcc_mult(const std::vector<double>& conf_matrix, int n, int K);
    }

    namespace OneHot {
        // Log loss for multiple categories with y (one hot encoded) and prob col major 
        double logloss_mult_onehot(const Dataframe& y, const Dataframe& prob);

        // Log likehood for mutliple categories with y (one hot encoded if K > 2) and prob col major 
        double logLikelihood_onehot(const Dataframe& y, const Dataframe& prob);

        // Log likehood null for multiple categories with y (one hot encoded if K > 2)
        double logLikelihood_null_onehot(const Dataframe& y);
    }
}