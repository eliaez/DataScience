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
        throw std::invalid_argument("Y and Y_pred need to have the same length");
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

   // Sort
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return prob[a] > prob[b];
    });

    double totalPos = std::count(y.begin(), y.end(), 1.0);
    double totalNeg = n - totalPos;

    double tp = 0, fp = 0, auc = 0;
    double prevFpr = 0, prevTpr = 0;
    double prevScore = -1;

    for (size_t i = 0; i < n; ++i) {
        if (prob[idx[i]] != prevScore && i > 0) {
            double fpr = fp / totalNeg;
            double tpr = tp / totalPos;
            auc += (fpr - prevFpr) * (tpr + prevTpr) / 2.0;
            prevFpr = fpr;
            prevTpr = tpr;
        }

        prevScore = prob[idx[i]];
        if (y[idx[i]] == 1.0) tp++; else fp++;
    }

    // Last point
    double fpr = fp / totalNeg;
    double tpr = tp / totalPos;
    auc += (fpr - prevFpr) * (tpr + prevTpr) / 2.0;

    return auc;
}

double mcc(double TP, double FN, double FP, double TN) {
    return ((TP * TN) - (FP * FN)) / std::sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
}

double mcc(const std::vector<double>& conf_matrix) {
    return mcc(conf_matrix[0], conf_matrix[1], conf_matrix[2], conf_matrix[3]);
}

double normal_pval(double z) {
    double cdf = 0.5 * (1.0 + std::erf(std::abs(z) / std::sqrt(2.0)));
    return 2.0 * (1.0 - cdf);
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

Dataframe fisher_mat(const Dataframe& x_const, const Dataframe& y_proba, size_t ref_class) {

    if (x_const.get_storage())
        throw std::invalid_argument("X must be col major");
    if (y_proba.get_storage())
        throw std::invalid_argument("Y_pred must be col major");

    size_t n = x_const.get_rows();
    size_t p = x_const.get_cols();
    size_t K = y_proba.get_cols();
    size_t K_ = K - 1;
    size_t dim  = K_ * p;

    // Only for non ref categories
    std::vector<size_t> cats;
    cats.reserve(K_);
    for (size_t k = 0; k < K; k++)
        if (k != ref_class) cats.push_back(k);

    std::vector<double> F(dim * dim, 0.0);
    for (size_t i = 0; i < K_; i++) {
        for (size_t l = i; l < K_; l++) {

            size_t ki = cats[i];
            size_t kl = cats[l];

            // Accumulate Block i,l : X^T * D_il * X
            std::vector<double> block(p * p, 0.0); 
            for (size_t obs = 0; obs < n; obs++) {
                double p_ik = y_proba.at(ki * n + obs);
                double p_il = y_proba.at(kl * n + obs);

                // Diagonal weight
                double d;
                if (ki == kl) {
                    d = p_ik * (1.0 - p_ik);
                    if (d < 1e-10) d = 0.0;
                } 
                else d = -p_ik * p_il;

                // Upper triangle of the outer product
                for (size_t j1 = 0; j1 < p; j1++) {

                    double xj1 = (j1 < p) ? x_const.at(j1*n + obs) : 1.0;
                    for (size_t j2 = j1; j2 < p; j2++) {

                        double xj2 = (j2 < p) ? x_const.at(j2*n + obs) : 1.0;
                        block[j1*p + j2] += xj1 * d * xj2;
                    }
                }
            }

            // Write block (i,l) and its symmetric counterparts into F (col-major)
            for (size_t j1 = 0; j1 < p; j1++) {
                for (size_t j2 = j1; j2 < p; j2++) {
                    
                    double val = block[j1*p + j2];
                    F[(l*p + j2)*dim + (i*p + j1)] = val;
                    F[(i*p + j1)*dim + (l*p + j2)] = val;

                    if (i != l) {
                        F[(i*p + j2)*dim + (l*p + j1)] = val;
                        F[(l*p + j1)*dim + (i*p + j2)] = val;
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < dim; i++) F[i*dim + i] += 1e-8;
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
    std::vector<double> se(p);
    for (size_t j = 0; j < p; j++) {

        double fisher = 0.0;
        for (size_t i = 0; i < n; i++) {
            double p_ik = y_pred.at(K * n + i);
            double xij = (j == 0) ? 1.0 : x.at((j - 1) * n + i);
            fisher += p_ik * (1.0 - p_ik) * xij * xij;
        }
        se[j] = (fisher > 0.0) ? std::sqrt(1.0 / fisher) : 0.0;
    }
    return se;
}

std::vector<double> stderr_coeff(const Dataframe& cov, int K, int p) {
    size_t tot_cols  = cov.get_cols();
    std::vector<double> se(p);
    for (size_t i = 0; i < p; i++) {
        size_t idx = K * p + i;
        se[i] = std::sqrt(cov.at(idx * tot_cols  + idx));
    }
    return se;
}

double mc_fadden(double loglikelihood_model, double loglikelihood_null) {
    return 1 - loglikelihood_model / loglikelihood_null;
}

double chi2_pval(double loglikelihood_model, double loglikelihood_null, double df) {
    boost::math::chi_squared dist(df);
    double chi2_stat =  -2 * (loglikelihood_null - loglikelihood_model);

    double pval = NAN;
    try {
        pval = 1 - boost::math::cdf(dist, chi2_stat);
    }
    catch (const std::exception& e) {
        std::cout << "LogLikelihood null > LogLikelihood model"
                << "Details: " << e.what() << std::endl;
    }
    return pval;
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
        if (K < 2) throw std::invalid_argument("Invalid K, y should be one-hot encoded:" + std::to_string(K));
        if (N != prob.get_rows()) throw std::invalid_argument("Incompatible sizes");

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