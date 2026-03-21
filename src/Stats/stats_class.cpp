#include <cmath>
#include <string>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Stats/stats_class.hpp"
#include <boost/math/distributions/chi_squared.hpp>

using namespace Utils;

namespace Stats_class {

std::vector<double> conf_matrix(const std::vector<double>& y, const std::vector<double>& y_pred) {
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

double logLikelihood(const std::vector<double>& y, const Dataframe& prob) {
    size_t N = y.size();
    return - Mult::logloss_mult(y, prob) * N;
}

double logLikelihood_null(const std::vector<double>& y, int K) {
    size_t N = y.size();

    // Count nk for each class
    std::vector<double> nk(K, 0.0);
    for (size_t i = 0; i < N; i++)
        nk[static_cast<int>(y[i])]++;

    double ll = 0.0;
    for (size_t k = 0; k < K; k++)
        if (nk[k] > 0)
            ll += nk[k] * std::log(nk[k] / N);
    return ll;
}

Dataframe fisher_mat(const Dataframe& x, const Dataframe& y_pred) {

    if (x.get_storage())
        throw std::invalid_argument("X must be col major");
    if (y_pred.get_storage())
        throw std::invalid_argument("Y_pred must be col major");

    size_t n = x.get_rows();
    size_t p = x.get_cols();
    size_t K = y_pred.get_cols();
    size_t m = p + 1;
    size_t dim = (K-1) * m;
    std::vector<double> F(dim * dim, 0.0);

    // For K = 2 we reduce to one col
    if (K == 1) {

        std::vector<double> sum(m, 0.0);
        for (size_t j1 = 0; j1 < m; j1++) {

            std::fill(sum.begin(), sum.end(), 0.0);
            for (size_t i = 0; i < n; i++) {
                
                double p_i = y_pred.at(i);
                double d   = p_i * (1.0 - p_i);
                double xid = x.at(j1*n + i) * d;
                for (size_t j2 = 0; j2 < m; j2++)
                    sum[j2] += xid * x.at(j2*n + i);
            }
            
            // F col-major, dim = m
            for (size_t j2 = 0; j2 < m; j2++)
                F[j2*m + j1] = sum[j2];
        }
        return {dim, dim, false, std::move(F)};
    }

    for (int k = 0; k < K-1; k++) {
        for (int l = k; l < K-1; l++) {

            // Block (k,l) : m x m
            std::vector<double> block(m * m, 0.0);

            for (int j1 = 0; j1 < m; j1++) {
                std::vector<double> sum(m, 0.0);

                for (int i = 0; i < n; i++) {
                    double p_ik = y_pred.at(k*n + i);
                    double p_il = y_pred.at(l*n + i);
                    double d    = (k == l) ? p_ik*(1.0 - p_ik) : -p_ik*p_il;
                    double xid  = x.at(j1*n + i) * d;

                    for (int j2 = 0; j2 < m; j2++)
                        sum[j2] += xid * x.at(j2*n + i);
                }

                // F col-major
                for (int j2 = 0; j2 < m; j2++) {
                    F[(l*m + j2)*dim + (k*m + j1)] = sum[j2];
                    if (k != l)
                        F[(k*m + j1)*dim + (l*m + j2)] = sum[j2];
                }
            }
        }
    }
    return {dim, dim, false, std::move(F)};
}

Dataframe cov_mat(Dataframe& fisher) {
    if (fisher.get_storage()) fisher.is_symmetric();
    return fisher.inv();
}

std::vector<double> stderr_coeff(const Dataframe& x, const Dataframe& y_pred, int K) {
    if (x.get_storage())
        throw std::invalid_argument("X must be col major");
    if (y_pred.get_storage())
        throw std::invalid_argument("Y_pred must be col major");

    size_t n = x.get_rows();
    size_t p = x.get_cols();
    size_t m = p + 1;
    std::vector<double> se(m);
    for (size_t j = 0; j < m; j++) {

        double fisher = 0.0;
        for (size_t i = 0; i < n; i++) {
            double p_ik = y_pred.at(K * n + i);
            double xij  = x.at(j * n + i);
            fisher += p_ik * (1.0 - p_ik) * xij * xij;
        }
        se[j] = std::sqrt(1.0 / fisher);
    }
    return se;
}

std::vector<double> stderr_coeff(const Dataframe& cov, int K) {
    size_t p = cov.get_cols();
    std::vector<double> se(p);
    for (size_t i = 0; i < p; i++) {
        se[i] = std::sqrt(cov.at(i * p + i));
    }
    return se;
}

std::vector<double> stderr_coeff(Dataframe& fisher, int K) {
    Dataframe cov = cov_mat(fisher);
    return stderr_coeff(cov, K);
}

double mc_fadden(double loglikehood_model, double loglikehood_null) {
    return 1 - loglikehood_model / loglikehood_null;
}

double chi2_pval(double loglikehood_model, double loglikehood_null, double df) {
    boost::math::chi_squared dist(df);
    double chi2_stat =  -2 * (loglikehood_null - loglikehood_model);
    return 1 - boost::math::cdf(dist, chi2_stat);
}

namespace Mult {
    std::vector<double> conf_matrix_mult(const std::vector<double>& y, const Dataframe& y_pred) {
        if (y.empty()) throw std::invalid_argument("Cannot calculate Confusion matrix of empty vector");
        if (y_pred.get_storage()) throw std::invalid_argument("Y_proba need to be col major");

        size_t n = y.size();
        if (n != y_pred.get_rows()) throw std::invalid_argument("Y and Y_pred need to have the same length");

        // Calc confusion matrix
        size_t K = y_pred.get_cols();
        std::vector<double> cm(K * K, 0.0);
        for (size_t i = 0; i < n; i++) {
            int true_class = static_cast<int>(y[i]);
            int pred_class = static_cast<int>(y_pred.at(i));
            cm[true_class * K + pred_class]++;
        }
        return cm;
    }

    double logloss_mult(const std::vector<double>& y, const Dataframe& prob) {
        
        // Tests
        size_t N = y.size();
        size_t K = prob.size() / N;
        if (prob.size() % K != 0) throw std::invalid_argument("Incompatible sizes");
        if (K < 2) throw std::invalid_argument("Invalid K:" + std::to_string(K));
        if (y.empty()) throw std::invalid_argument("Cannot calculate LogLoss of empty vector");
        if (prob.get_storage()) throw std::invalid_argument("Y_pred need to be col major");

        double sum = 0.0;
        for (size_t i = 0; i < N; i++) {
            int label = static_cast<int>(y[i]);
            sum += std::log(prob.at(label * N + i) + 1e-15);
        }
        return -sum / N;
    }

    std::vector<double> roc_auc_mult(const std::vector<double>& y, const Dataframe& prob) {
        if (prob.get_storage()) throw std::invalid_argument("Y_proba need to be col major");

        size_t n = y.size();
        size_t K = prob.get_cols();
        std::vector<double> aucs(K);
        for (int k = 0; k < K; k++) {
            
            std::vector<double> y_(n);
            std::vector<double> prob_k(n);
            for (size_t i = 0; i < n; i++) {
                y_[i] = (y[i] == k) ? 1.0 : 0.0;
                prob_k[i] = prob.at(k * n + i);
            }
            aucs[k] = roc_auc(y_, prob_k);
        }
        return aucs;
    }

    double mcc_mult(const std::vector<double>& conf_matrix, int n, int K) {
        double sum_diag = 0;
        double dot_rc = 0;
        double sum_r2 = 0, sum_c2 = 0;
        for (size_t k = 0; k < K; k++) {

            sum_diag += conf_matrix[k * K + k];
            double rk = 0, ck = 0;
            for (size_t j = 0; j < K; j++) {
                rk += conf_matrix[k * K + j];
                ck += conf_matrix[j * K + k];
            }

            dot_rc += rk * ck;
            sum_r2 += rk * rk;
            sum_c2 += ck * ck;
        }
        double mcc_num = n * sum_diag - dot_rc;
        double mcc_den = std::sqrt((n*n - sum_r2) * (n*n - sum_c2));
        return (mcc_den == 0) ? 0.0 : mcc_num / mcc_den;
    }
}

namespace OneHot {
    double logloss_mult_onehot(const Dataframe& y, const Dataframe& prob) {
        if (y.get_storage()) throw std::invalid_argument("Y need to be col major");
        if (prob.get_storage()) throw std::invalid_argument("Y_pred need to be col major");

        size_t N = y.get_rows();
        size_t K = y.get_cols();
        size_t n = N * K;
        if (K < 2) throw std::invalid_argument("Invalid K:" + std::to_string(K));
        if (N != prob.get_rows()) throw std::invalid_argument("Incompatible sizes");

        double max_y = *std::max_element(y.get_data().begin(), y.get_data().end());
        if (max_y > 1) throw std::invalid_argument("y should be one-hot encoded");

        // Log loss
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            if (y.at(i) == 1)
                sum += y.at(i) * std::log(prob.at(i) + 1e-15);
        }
        return - sum / N;
    }

    double logLikelihood_onehot(const Dataframe& y, const Dataframe& prob) {
        size_t N = y.get_rows();
        size_t K = y.get_cols();
        return - logloss_mult_onehot(y, prob) * N;
    }

    double logLikelihood_null_onehot(const Dataframe& y) {
        size_t N = y.get_rows();
        size_t K = y.get_cols();
        if (y.get_storage()) throw std::invalid_argument("Y need to be col major");
        
        // Count nk for each class
        std::vector<double> nk(K, 0.0);
        for (size_t i = 0; i < N; i++) {
            for (size_t k = 0; k < K; k++) {
                nk[k] += y.at(k * N + i);  // One-hot
            }
        }
        
        double ll = 0.0;
        for (size_t k = 0; k < K; k++) {
            if (nk[k] > 0)
                ll += nk[k] * std::log(nk[k] / N);
        }
        return ll;  
    }
}
}