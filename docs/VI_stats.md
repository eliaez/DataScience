# VI - Statistical Functions

Statistical toolkit for regression analysis, model validation, and inference. Provides both basic descriptive statistics (`mean`, `var`, `cov`, ...) and advanced diagnostic tests with optimized backends (`Naive` and `AVX2` by default). All the following functions are automatically called during the regression pipeline but can also be used as standalone utilities.

## Regression Diagnostics
Once the model is estimated, assessing its reliability and quality is essential. This section covers the functions implemented to evaluate both the statistical validity of the estimators and the overall goodness-of-fit of the model.

### Covariance Matrix
Computes the variance-covariance matrix of the β estimators, used to derive standard 
errors, t-statistics and confidence intervals for inference (`cov_beta_OLS`):
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
std::vector<double> cov_beta = Stats::cov_beta_OLS(X, XtXinv, residuals, "HC3");
```

**Interpretation**: 
- Diagonal values are the variance of each β estimator → `SE(βᵢ) = √Var(βᵢ)`
- Off-diagonal values are the covariance between estimators, non-zero values indicate potential multicollinearity

### Model Quality Metrics
Computes standard goodness-of-fit and prediction error metrics to evaluate model performance:
- **`rsquared()`**: The R² shows the proportion of total variance in y explained by the model, ranging from 0 to 1

- **`radjusted()`**: The adjusted R² penalizes model complexity by accounting for the number of predictors (including intercept), thus preventing overfitting

- **`mae()`, `mse()`, `rmse()`**: The **Mean Absolute Error** is used to measure the average prediction error (robust to outliers), the **Mean Squared Error** to penalize large errors more heavily and the **Root Mean Squared Error** for an easier interpretation.

```cpp
// Goodness-of-fit
double r2     = Stats::rsquared(y, y_pred);
double r2_adj = Stats::radjusted(y, y_pred, p);

// Prediction error
double mae  = Stats::mae(y, y_pred);
double mse  = Stats::mse(y, y_pred);
double rmse = Stats::rmse(y, y_pred);
```

**Interpretation**:
- R² and adjusted R² measure how well the model fits the data, higher is better.
- MAE, MSE and RMSE measure prediction error, lower is better. Prefer RMSE for interpretability, MAE when robustness to outliers matters.

## Hypothesis Testing

### Fisher Test
Tests global model significance under H₀: all coefficients equal zero. A low p-value rejects H₀, indicating that the model as a whole explains a statistically significant portion of variance. Supports robust covariance types (HC3, HAC, cluster) via a Wald test alternative.

```cpp
// Fisher test with df1 = p and df2 = n - p - 1 and cov_type : classical, HC3,...
double F_stat = Stats::fisher_test(r2, df1, df2, beta_est, cov_beta, "HC3");
double F_pval = Stats::fisher_pvalue(F_stat, df1, df2); 
// Low p-value → reject H₀, model is globally significant
```

### Student Test
Tests individual coefficient significance under H₀: βⱼ = 0. A low p-value rejects H₀, indicating that the predictor has a statistically significant individual effect on y independently of the other predictors.

```cpp
std::vector<double> stderr = Stats::stderr_b(cov_beta);
std::vector<double> p_vals = Stats::student_pvalue(t_stats);
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
double rho = Stats::durbin_watson_test(residuals);
```
<br>

**`breusch_pagan_test()`** detects heteroskedasticity by testing whether error variance is constant across observations. Under H₀ (homoskedasticity), the test statistic follows a χ²(p) distribution and returns a p-value: p < 0.05 → reject H₀, switch to robust standard errors (HC3/HAC).

```cpp
double p_bp = Stats::breusch_pagan_test(X, residuals);
```

### Multicollinearity
The **Variance Inflation Factor** measures how much the standard error of βⱼ is inflated due to correlation with other predictors: `VIFⱼ = 1 / (1 - R²ⱼ)` where R²ⱼ comes from regressing Xⱼ on all other predictors. A VIF below 5 is acceptable, while a VIF above 10 signals severe multicollinearity, suggesting to remove or combine correlated features.

```cpp
std::vector<double> vif = Stats::VIF(X);
// or with GLS
std::vector<double> vif = Stats::VIF(X, omega);
```
<br>

**To test it yourself**, you can also check the corresponding files: 
- [**stats.hpp**](/include/Stats/stats_reg.hpp)
- [**stats.cpp**](/src/Stats/stats_reg.cpp)
- [**Test folder**](/tests/)

#### Note: 
By default, the backend used will be the best performing one among the three customized implementations, excluding MKL and Eigen libraries. Moreover, to compile MKL, the MSVC option is available to avoid any issue with MinGW-GCC.

To read the next part: [**V - Regressions**](/docs/V_regressions.md).