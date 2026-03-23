#include <iomanip>
#include <iostream>
#include "Data/Data.hpp"
#include "Utils/Utils.hpp"
#include "Stats/stats_reg.hpp"
#include "Stats/stats_class.hpp"
#include "Validation/Validation.hpp"
#include "Models/Supervised/Classification/LogisticReg.hpp"

using namespace Utils;

namespace Class {

Dataframe LogisticRegression::fit_without_stats(const Dataframe& x, const Dataframe& y) {
    
    // Tests
    basic_verif(x);
    basic_verif(y);
    if (x.get_storage()) {
        throw std::invalid_argument("Need x col-major");
    }

    nb_categories(y);
    if (y.get_cols() != nb_cats && nb_cats > 2 && y.get_cols() != 1) {
        throw std::invalid_argument("For Y, either input Y col major with nb of cols = nb of categories or only one col");
    }
    else if (y.get_cols() == nb_cats && nb_cats > 2 && y.get_storage()) {
        throw std::invalid_argument("For Y, if you input Y with nb of cols = nb of categories then Y need to be col major");
    }

    size_t n = x.get_rows();
    size_t p = x.get_cols();

    // Copy our data 
    std::vector<double> x_v = x.get_data();
    
    // Insert an unit col to get intercept value
    x_v.insert(x_v.begin(), n, 1.0);

    // Need X
    Dataframe X = {n, p+1, false, std::move(x_v)};
    Dataframe X_T = ~X;
    X_T.change_layout_inplace();

    // Our vector W
    std::vector<double> w((p+1) * nb_cats, 0.0);
    Dataframe W = {p+1, nb_cats, false, std::move(w)};

    // Gradient Descent
    int idx = 0;
    Dataframe Y_ = y;
    bool keep_cond = true;
    if (nb_cats > 1 && y.get_cols() == 1) Y_.OneHot(0);
    double loss = std::numeric_limits<double>::infinity();
    double old_loss = std::numeric_limits<double>::infinity();
    while (keep_cond && idx < max_iter_) {

        // Softmax
        std::vector<double> y_v = softmax(X, W);
        Dataframe Y_pred = Dataframe(n, nb_cats, false, std::move(y_v));

        // Calculate our gradient
        std::vector<double> gradient_v = (X_T * (Y_pred - Y_)).get_data();
        gradient_v = mult(gradient_v, 1.0 / n);

        // Cost function 
        old_loss = loss;
        loss = Stats_class::OneHot::logloss_mult_onehot(Y_, Y_pred);

        // Add penality to our loss and gradient
        if (penality_ == 1.0) {
            for (size_t i = 0; i < (p+1)*nb_cats; i++) {

                // To exclude w0
                if (i % (p+1) == 0) continue;
                loss += std::abs(W.at(i)) / C_;
                gradient_v[i] += (W.at(i) > 0 ? 1.0 : -1.0) / C_;
            }
        }
        else if (penality_ == 2.0) {
            for (size_t i = 0; i < (p+1)*nb_cats; i++) {

                // To exclude w0
                if (i % (p+1) == 0) continue;
                loss += W.at(i) * W.at(i) / (2 * C_);
                gradient_v[i] += W.at(i) / C_;
            }
        }
        else if (penality_ == 1.5) {
            for (size_t i = 0; i < (p+1)*nb_cats; i++) {
                
                // To exclude w0
                if (i % (p+1) == 0) continue;
                double w_i = W.at(i);
                loss += (l1_ratio_ * std::abs(w_i) + (1.0 - l1_ratio_) / 2.0 * w_i * w_i) / C_;
                gradient_v[i] += (l1_ratio_ * (w_i > 0 ? 1.0 : -1.0) + (1.0 - l1_ratio_) * w_i) / C_;
            }
        } 
        else if (penality_ == 0.0) {}
        else {
            throw std::invalid_argument("Unknown penality: " + std::to_string(penality_));
        }

        // Finishing calculating our gradient
        gradient_v = mult(gradient_v, learning_r_);
        Dataframe gradient = {p+1, nb_cats, false, std::move(gradient_v)};
        
        // Updating our W
        W = W - gradient;

        // Testing convergence of cost
        if (std::abs(loss - old_loss) / (std::abs(old_loss) + 1e-10) < tol_) { // Threshold
            break;
        }
        idx++;
    }
    if (max_iter_ == idx) 
        std::cout << "Max_iter reached, you might need to change learning_rate or scale your data\n" << std::endl;

    // Results
    coeffs = W.get_data();
    is_fitted = true;

    return X;
}

/*
void LogisticRegression::optimal_lambda(double start, double end, int nb, const Dataframe& x, const Dataframe& y) {
    std::vector<double> path(nb);
    double log_min = log(start);
    double log_max = log(end);
    double step = (log_max - log_min) / (nb - 1);

    for (int i = 0; i < nb; i++) {
        path[i] = exp(log_min + i * step);
    }

    std::vector<std::vector<double>> param_grid = {path};
    Validation::GSres res = Validation::GSearchCV(this, x, y, param_grid);

    lambda_ = res.best_params[0];
}*/

std::unique_ptr<ClassificationBase> LogisticRegression::create(const std::vector<double>& params) {
    return std::make_unique<LogisticRegression>(params[0], params[1]);
}

void LogisticRegression::compute_stats(const Dataframe& x, Dataframe& x_const, const Dataframe& y) {
    
    size_t n = x.get_rows();
    size_t p = x.get_cols();
    size_t K = nb_cats == 2 ? 1.0 : nb_cats;
    
    // Predict 
    std::vector<double> y_proba = predict_proba(x);
    Dataframe Y_proba = {n, nb_cats, false, std::move(y_proba)};

    std::vector<double> y_pred = predict(x);
    Dataframe Y_pred = {n, K, false, std::move(y_pred)};

    //auto save_vec = [](const std::string& path, const std::vector<double>& v) {
    //    std::ofstream f(path);
    //    for (auto& x : v) f << std::setprecision(10) << x << "\n";
    //};
    //
    //save_vec("C:/Users/romai/y_pred.csv", Y_proba.get_data());

    // Fisher matrix
    Dataframe fisher = Stats_class::fisher_mat(x_const, Y_proba);

    auto data = fisher.get_data();
    bool is_null = std::all_of(data.begin(), data.end(), [](double v){ return std::abs(v) < 1e-10; });
    if (is_null) {
        std::cout << "Fisher matrix is null"
                << "stderr and z-stats unavailable. "
                << "Consider regularization or checking class balance.\n";
    }

    // Covariance matrix

    Dataframe cov_mat;
    try {
        cov_mat = Stats_class::cov_mat(fisher);
    }
    catch (const std::exception& e) {
        std::cout << "Fisher matrix is singular, stderr and z-stats unavailable. "
                << "Consider regularization or checking class balance.\n"
                << "Details: " << e.what() << std::endl;
        std::vector<double> vect_null((nb_cats - 1) * (p + 1) * (nb_cats - 1) * (p + 1), 0.0);
        cov_mat = {(nb_cats - 1) * (p + 1), (nb_cats - 1) * (p + 1), false, std::move(vect_null)};
    }
    

    // Confusion matrix
    std::vector<double> conf_matrix;
    if (nb_cats == 2) conf_matrix = Stats_class::conf_matrix(y.get_data(), Y_pred.get_data());
    else conf_matrix = Stats_class::Mult::conf_matrix_mult(y.get_data(), Y_pred);

    // Roc Auc 
    std::vector<double> y_proba_bin(n);
    for (int i = 0; i < n; i++)
        y_proba_bin[i] = y_proba[i * 2 + 1];
    std::vector<double> roc_auc;
    if (nb_cats == 2) roc_auc.push_back(Stats_class::roc_auc(y.get_data(), y_proba_bin));
    else roc_auc = Stats_class::Mult::roc_auc_mult(y.get_data(), Y_proba);

    // If we have not the cols name
    std::vector<std::string> headers(p+1, "");
    headers[0] = "Intercept";
    if (x.get_headers().empty()) {
        for (size_t i = 1; i < p+1; i++) headers[i] = "c" + std::to_string(i);
    }
    else {
        headers = {"Intercept"};
        headers.insert(headers.end(), x.get_headers().begin(), x.get_headers().end());
    }

    if (nb_cats == 2) {
        // Coeff stats
        CoeffStats c;
        c.category = "Class 0 vs Class 1";
        c.stderr_beta = Stats_class::stderr_coeff(cov_mat, 0, p+1);
        for (size_t i = 0; i < p+1; i++) {
            c.name.push_back(headers[i]);
            double beta = coeffs[i];
            c.beta.push_back(beta);
            c.odds_ratio.push_back(std::exp(beta));
            c.z_stat.push_back(c.stderr_beta[i] ==  0.0 ? NAN : beta / c.stderr_beta[i]);
            c.p_value.push_back(Stats::normal_cdf(c.z_stat.back()));
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

        double df = p + 1;
        double logL = Stats_class::logLikelihood(y.get_data(), Y_proba);
        double logLnull = Stats_class::logLikelihood_null(y.get_data(), nb_cats);
        double accuracy = (TP + TN) / n;

        gen_stats.push_back(logL);
        gen_stats.push_back(Stats::Regularized::AIC(df, logL));
        gen_stats.push_back(Stats::Regularized::BIC(df, logL, n));
        gen_stats.push_back(Stats_class::mc_fadden(logL, logLnull));
        gen_stats.push_back(Stats_class::chi2_pval(logL, logLnull, df));
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
            c.category = "Class " + std::to_string(cat) + " vs Class " +  std::to_string(ref_class_);
            c.stderr_beta = Stats_class::stderr_coeff(cov_mat, cat, p+1);
            for (size_t i = 0; i < p+1; i++) {
                c.name.push_back(headers[i]);

                double displayed_beta = coeffs[cat * (p+1) + i] - coeffs[ref_class_ * (p+1) + i];
                c.beta.push_back(displayed_beta);
                c.odds_ratio.push_back(std::exp(displayed_beta));
                c.z_stat.push_back(c.stderr_beta[i] ==  0.0 ? NAN : displayed_beta / c.stderr_beta[i]);
                c.p_value.push_back(Stats::normal_cdf(c.z_stat[i]));
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

        double df = (nb_cats - 1) * (p + 1); // Without penality correction 
        double logL = Stats_class::logLikelihood(y.get_data(), Y_proba);
        double logLnull = Stats_class::logLikelihood_null(y.get_data(), nb_cats);

        // Accuracy
        double count = 0.0;
        for (size_t i = 0; i < K; i++) count += conf_matrix[i * K + i];
        double accuracy = count / n;

        // Save general stats
        gen_stats.push_back(logL);
        gen_stats.push_back(Stats::Regularized::AIC(df, logL));
        gen_stats.push_back(Stats::Regularized::BIC(df, logL, n));
        gen_stats.push_back(Stats_class::mc_fadden(logL, logLnull));
        gen_stats.push_back(Stats_class::chi2_pval(logL, logLnull, df));
        gen_stats.push_back(Stats_class::Mult::mcc_mult(conf_matrix, n, K));
        gen_stats.push_back(accuracy);
        gen_stats.push_back(mean(precision));
        gen_stats.push_back(mean(recall));
        gen_stats.push_back(mean(specificity));
        gen_stats.push_back(mean(f1));
        gen_stats.push_back(mean(roc_auc));
    }
}

void LogisticRegression::summary(bool detailled) const {

    std::cout << "\n=== Classification SUMMARY ===\n\n";

    // Global Stats
    std::cout << "Log-Likelihood = " << gen_stats[0] << "\n";
    std::cout << "AIC            = " << gen_stats[1] << "\n";
    std::cout << "BIC            = " << gen_stats[2] << "\n";
    std::cout << "McFadden R2    = " << gen_stats[3] << "\n";
    std::cout << "Chi2 p-value   = " << gen_stats[4] << "\n";
    std::cout << "MCC            = " << gen_stats[5] << "\n";
    std::cout << "Accuracy       = " << gen_stats[6] << "\n\n";

    if (detailled) {
        std::cout << "  Precision  = " << gen_stats[7] << "\n";
        std::cout << "  Recall     = " << gen_stats[8] << "\n";
        std::cout << "  Specificity= " << gen_stats[9] << "\n";
        std::cout << "  F1         = " << gen_stats[10] << "\n";
        std::cout << "  ROC AUC    = " << gen_stats[11] << "\n\n";
    }

    // For each class
    for (const auto& stat : coeff_stats) {

        std::cout << "--- " << stat.category << " ---\n";

        // Metrics
        std::cout << "  Precision=" << std::fixed << std::setprecision(4) << stat.gen_stats[0]
                  << "  Recall="    << stat.gen_stats[1]
                  << "  Specificity=" << stat.gen_stats[2]
                  << "  F1="        << stat.gen_stats[3]
                  << "  ROC AUC="   << stat.gen_stats[4] << "\n\n";

        // Coefficients table
        std::cout << std::left  << std::setw(15) << "Coefficient"
                  << std::right << std::setw(12) << "Beta"
                  << std::setw(12) << "Odds Ratio"
                  << std::setw(12) << "Stderr"
                  << std::setw(12) << "z-stat"
                  << std::setw(12) << "p-value"
                  << std::setw(5)  << "Sig" << "\n";
        std::cout << std::string(85, '-') << "\n";

        for (size_t i = 0; i < stat.name.size(); i++) {
            std::cout << std::left  << std::setw(15) << stat.name[i]
                      << std::right << std::fixed << std::setprecision(4)
                      << std::setw(12) << stat.beta[i]
                      << std::setw(12) << stat.odds_ratio[i]
                      << std::setw(12) << stat.stderr_beta[i]
                      << std::setw(12) << stat.z_stat[i]
                      << std::setw(12) << stat.p_value[i]
                      << "  " << stat.significance(i) << "\n";
        }
        std::cout << "\n";
    }

    std::cout << "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1\n\n";
    std::cout << std::endl;
}
}