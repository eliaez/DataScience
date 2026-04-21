# VI - Statistical Functions

Statistical toolkit for regression analysis, model validation, and inference. Provides both basic descriptive statistics (`mean`, `var`, `cov`, ...) and advanced diagnostic tests with optimized backends (`Naive` and `AVX2` by default). The module also extends to time series analysis through `Stats_TS` by enabling automatic `ARIMA` and `SARIMA` parameters identification, and to classification through `Stats_class` which covers everything from binary confusion metrics to multi-class inference.  All the following functions are automatically called during the regression pipeline but can also be used as standalone utilities.

## 1 - Regression Diagnostics
Once the model is estimated, assessing its reliability and quality is essential. This section covers the functions implemented to evaluate both the statistical validity of the estimators and the overall goodness-of-fit of the model.

### Covariance Matrix
Computes the variance-covariance matrix of the β estimators, used to derive standard 
errors, t-statistics and confidence intervals for inference (`Stats::OLS::cov_beta`):
- **`classical`**: Assumes standard OLS conditions (homoskedasticity, no autocorrelation)

- **`HC3`**: Implements White's Heteroskedasticity-Consistent (HC) estimator which provides an  adjustment for non-constant error variance without assuming a specific structure

- **`HAC`**: Heteroskedasticity and Autocorrelation Consistent (HAC) Newey-West estimator which is robust to both autocorrelation and heteroskedasticity, typically used for time series data

- **`cluster`**: Accounts for within-group correlation by clustering standard errors which is
suitable when observations are grouped (individuals within firms)

- **`GLS`**: Generalized Least Squares (GLS) incorporates a known variance structure Ω

```cpp
// After fitting a regression model
std::vector<double> residuals = Stats::get_residuals(y, y_pred);

// Compute Covariance Matrix 
std::vector<double> cov_beta = Stats::OLS::cov_beta(X, XtXinv, residuals, "HC3");
```

**Interpretation**: 
- Diagonal values are the variance of each β estimator → `SE(βᵢ) = √Var(βᵢ)`
- Off-diagonal values are the covariance between estimators, non-zero values indicate potential multicollinearity

### Model Quality Metrics
Computes standard goodness-of-fit and prediction error metrics to evaluate model performance:
- **`rsquared()`**: The R² shows the proportion of total variance in y explained by the model, ranging from 0 to 1

- **`radjusted()`**: The adjusted R² penalizes model complexity by accounting for the number of predictors (including intercept), thus preventing overfitting

- **`mae()`, `mse()`, `rmse()`**: The **Mean Absolute Error** is used to measure the average prediction error (robust to outliers), the **Mean Squared Error** to penalize large errors more heavily and the **Root Mean Squared Error** for an easier interpretation

- **`logLikelihood()`**: Computes the log-likelihood of the model, used for information criteria and model comparison.

```cpp
// Goodness-of-fit
double r2     = Stats::rsquared(y, y_pred);
double r2_adj = Stats::radjusted(r2, n, p);

// Prediction error
double mae  = Stats::mae(y, y_pred);
double mse  = Stats::mse(y, y_pred);
double rmse = Stats::rmse(y, y_pred);

// Log-likelihood
double ll = Stats::logLikelihood(y, y_pred);
```

**Interpretation**:
- R² and adjusted R² measure how well the model fits the data, higher is better.
- MAE, MSE and RMSE measure prediction error, lower is better. Prefer RMSE for interpretability, MAE when robustness to outliers matters.
- Log-likelihood measures how well the model fits the data probabilistically, higher is better.

### Information Criteria (Regularized Models)
For regularized models (`Ridge`, `Lasso`, ...), the effective degrees of freedom replace the standard parameter count. Our methods, `Stats::Regularized::AIC` and `Stats::Regularized::BIC` will penalize model complexity to balance fit and parsimony:

- **`AIC`**: Akaike Information Criterion, penalizes by `2 × df`

- **`BIC`**: Bayesian Information Criterion, penalizes more strongly as `n` grows via `log(n) × df`

```cpp
double aic = Stats::Regularized::AIC(effective_df, loglikehood);
double bic = Stats::Regularized::BIC(effective_df, loglikehood, n);
```

**Interpretation:** Both `AIC` and `BIC` compare models by balancing goodness-of-fit against complexity. A lower score indicates a better trade-off between the two. The key difference lies in how strongly each penalizes complexity: `BIC` tends to select more parsimonious models than `AIC` for large `n`.

## 2 - Hypothesis Testing

### Fisher Test
Tests global model significance under H₀: all coefficients equal zero. A low p-value rejects H₀, indicating that the model as a whole explains a statistically significant portion of variance. Supports robust covariance types (HC3, HAC, cluster) via a Wald test alternative.

```cpp
// Fisher test with df1 = p and df2 = n - p - 1 and cov_type : classical, HC3,...
double F_stat = Stats::OLS::fisher_test(r2, df1, df2, beta_est, cov_beta, "HC3");
double F_pval = Stats::OLS::fisher_pvalue(F_stat, df1, df2); 
// Low p-value → reject H₀, model is globally significant
```

### Student Test
Tests individual coefficient significance under H₀: βⱼ = 0. A low p-value rejects H₀, indicating that the predictor has a statistically significant individual effect on y independently of the other predictors.

```cpp
std::vector<double> stderr = Stats::OLS::stderr_b(cov_beta);
std::vector<double> p_vals = Stats::OLS::student_pvalue(t_stats);
// Low p-value → reject H₀, coefficient is individually significant
```

### Residual Analysis

**`residuals_stats()`** provides a distribution summary (mean, Q1, median, Q3, max absolute value). To interpret the values; a mean close to zero confirms unbiased predictions, while Q1/Q3 symmetry suggests normally distributed errors.

```cpp
std::vector<double> res_stats = Stats::residuals_stats(residuals);
```
<br>

**`durbin_watson_test()`** detects serial autocorrelation in residuals. The statistic lies in [0, 4] where 2 indicates no autocorrelation. Returns the correlation coefficient `ρ = 1 − DW/2` where ρ > 0.25 indicates positive autocorrelation, meaning standard errors are underestimated and HAC should be used, while ρ ≈ 0 confirms no autocorrelation and OLS standard errors remain valid.

```cpp
double rho = Stats::OLS::durbin_watson_test(residuals);
```
<br>

**`breusch_pagan_test()`** detects heteroskedasticity by testing whether error variance is constant across observations. Under H₀ (homoskedasticity), the test statistic follows a χ²(p) distribution and returns a p-value: p < 0.05 → reject H₀, switch to robust standard errors (HC3/HAC).

```cpp
double p_bp = Stats::breusch_pagan_test(X, residuals);
```

### Multicollinearity
The **Variance Inflation Factor** measures how much the standard error of βⱼ is inflated due to correlation with other predictors: `VIFⱼ = 1 / (1 - R²ⱼ)` where R²ⱼ comes from regressing Xⱼ on all other predictors. A VIF below 5 is acceptable, while a VIF above 10 signals severe multicollinearity, suggesting to remove or combine correlated features.

```cpp
std::vector<double> vif = Stats::OLS::VIF(X);
// or with GLS
std::vector<double> vif = Stats::OLS::VIF(X, omega);
```

## 3 -  Time Series -  `ARIMA`/`SARIMA` Parameters Detection

Beyond static regression, the `Stats_TS` module extends the toolkit to time series analysis by automating the identification of `ARIMA` and `SARIMA` model orders through a combination of classical statistical tests:

```cpp
std::vector<int> params = Stats_TS::detect_ARIMA(y);
// Returns { p, d, q }

std::vector<int> params = Stats_TS::detect_SARIMA(y);
// Returns { p, d, q, P, D, Q, s, seasonality} (0 or 1 for false/true for seasonality)
```

**Interpretation**:
- `p`, `d`, `q`: AR order, differencing order and MA order
- `P`, `D`, `Q`: their seasonal counterparts
- `s`: detected seasonality period (12 for monthly data, 4 for quarterly)

### Detection process 
#### 1 - Stationarity

The first step in any `ARIMA` pipeline is to assess stationarity. **`ADF_test()`** checks whether the series needs to be differenced by testing for the presence of a unit root. The result is then compared against a threshold computed by **`critical_value_MacKinon()`**:

```cpp
double adf_stat = Stats_TS::ADF_test(y);
double cv       = Stats_TS::critical_value_MacKinon(y.size());
// adf_stat < cv → series is stationary -> d = 0
// adf_stat >= cv → differencing required -> d += 1
```

#### 2 - Seasonality Detection

Then, the seasonality is investigated through **`Acf_s()`** to identify the dominant seasonal period `s` which is then validated by **`Kruskal_Wallis()`** to confirm statistically significant seasonal differences across groups. **`Fft()`** subsequently refines this estimate by analyzing the frequency domain, all on the detrended series:

```cpp
std::vector<double> y_detrend = Stats_TS::linear_detrend(y);

int  s           = Stats_TS::Acf_s(y_detrend);
bool is_seasonal = Stats_TS::Kruskal_Wallis(y_detrend, s);
int  s_refined   = Stats_TS::Fft(y_detrend);
```

#### 3 - ACF & PACF

With stationarity and seasonality established, the remaining orders are identified through autocorrelation analysis. `Acf()` determines `q` from the cutoff lag, `Pacf()` determines `p` via the Durbin-Levinson algorithm (seasonal counterparts exist for `P`, `Q`):

```cpp
int q = Stats_TS::Acf(y);
int p = Stats_TS::Pacf(y);
```

**Note**: a grid search over (p, q) minimizing AIC/BIC would be more exhaustive at the cost of higher computational overhead.
<br>

## 4 - Classification Diagnostics (`Stats_class`)

Once a classifier is trained, evaluating its performance goes well beyond simple accuracy. The `Stats_class` namespace provides a complete suite of binary and multi-class metrics to assess coefficient significance.

### Confusion Matrix & Core Metrics

The foundation of binary classification evaluation is the confusion matrix, computed via **`conf_matrix()`** which returns `{TP, FN, FP, TN}` for a binary outcome. From these four values, the full set of standard metrics follows naturally: **`accuracy()`**, **`precision()`**, **`recall()`**,  **`specificity()`** and **`f1()`**.

```cpp
// Compute confusion matrix
std::vector<double> cm = Stats_class::conf_matrix(y, y_pred);
// cm = {TP, FN, FP, TN}

double acc  = Stats_class::accuracy(cm);
double prec = Stats_class::precision(cm[0], cm[2]);   // TP, FP
double rec  = Stats_class::recall(cm[0], cm[1]);      // TP, FN
double spec = Stats_class::specificity(cm[3], cm[2]); // TN, FP
double f1   = Stats_class::f1(prec, rec);
```

### Discrimination Metrics

**`roc_auc()`** evaluates the classifier's ability to rank positive examples above negative ones, regardless of any fixed threshold. Complementing this, **`mcc()`** (the Matthews Correlation Coefficient) provides a single balanced score that accounts for all four cells of the confusion matrix simultaneously (making it particularly reliable when classes are heavily imbalanced):

```cpp
// Threshold-independent discrimination
double auc = Stats_class::roc_auc(y, prob);

// Matthews Correlation Coefficient
double m = Stats_class::mcc(cm);
// or directly from raw counts
double m = Stats_class::mcc(TP, FN, FP, TN);
```

**Interpretation**: An AUC of 0.5 corresponds to a random classifier while 1.0 indicates perfect discrimination. MCC ranges from −1 (inverse predictions) to +1 (perfect predictions) with 0 indicating random chance.

### Multi-class Extension (`Stats_class::Mult` & `Stats_class::OneHot`)

When the number of classes exceeds two, the binary metrics generalize through the `Mult` namespace:

```cpp
// K×K confusion matrix (row-major), y_pred col-major Dataframe
std::vector<double> cm_k = Stats_class::Mult::conf_matrix_mult(y, y_pred, K);

// Log loss and multi-class AUC, prob col-major (one column per class)
double loss = Stats_class::Mult::logloss_mult(y, prob);
std::vector<double> aucs = Stats_class::Mult::roc_auc_mult(y, prob);

// Generalized MCC over K classes
double mcc_k = Stats_class::Mult::mcc_mult(cm_k, n, K);
```

When labels are one-hot encoded rather than integer-indexed, the `OneHot` namespace provides equivalent functions that operate directly on the encoded matrix:

```cpp
// y and prob are col-major Dataframes (N×K)
double loss_oh = Stats_class::OneHot::logloss_mult_onehot(y_onehot, prob);
double ll_oh   = Stats_class::OneHot::logLikelihood_onehot(y_onehot, prob);
double ll_null = Stats_class::OneHot::logLikelihood_null_onehot(y_onehot);
```

### Goodness-of-Fit & Model Comparison

Analogously to R² in linear regression, **`mc_fadden()`** quantifies the classification model's explanatory power relative to a null intercept-only model. Moreover, to formally test whether the improvement over the null is statistically significant, we use **`chi2_pval()`**:

```cpp
double ll_model = Stats_class::logLikelihood(y, prob);
double ll_null  = Stats_class::logLikelihood_null(y, K);

// McFadden R²
double r2_mf = Stats_class::mc_fadden(ll_model, ll_null);

// Likelihood-ratio test, df = (K-1) * p (p without intercept)
double pval = Stats_class::chi2_pval(ll_model, ll_null, df);
```

**Interpretation**: McFadden R² values above 0.2 are generally considered good and above 0.4 excellent, though they are not directly comparable to OLS R². 

### Coefficient Inference

Standard errors are derived from the Fisher information matrix (`fisher_mat()`), inverted into a covariance matrix (`cov_mat()`), from which `stderr_coeff()` and `normal_pval()` can be calculated:

```cpp
// Fisher information matrix, then covariance
Dataframe fisher = Stats_class::fisher_mat(x_const, y_proba, ref_class);
Dataframe cov    = Stats_class::cov_mat(fisher);

// Standard errors for class K (p features including intercept)
std::vector<double> se = Stats_class::stderr_coeff(cov, K, p);

// Or directly from design matrix and predicted probabilities
std::vector<double> se = Stats_class::stderr_coeff(x_const, y_pred, K);

// Two-tailed p-values from z-scores
double z    = beta[j] / se[j];
double pval = Stats_class::normal_pval(z);
```
<br>

**To test it yourself**, you can also check the corresponding files:
- [**stats.hpp**](/include/Stats/stats_reg.hpp)
- [**stats.cpp**](/src/Stats/stats_reg.cpp)
- [**Time_series.hpp**](/include/Stats/Time_series.hpp)
- [**Time_series.cpp**](/src/Stats/Time_series.cpp)
- [**stats_class.hpp**](/include/Stats/stats_class.hpp)
- [**stats_class.cpp**](/src/Stats/stats_class.cpp)
- [**Test folder**](/tests/)

To read the next part: [**V - Preprocessing**](/docs/V_preprocessing.md).