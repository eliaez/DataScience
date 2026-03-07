# VII - Regressions

Regression analysis is at the core of this library. All models implemented here share a common interface through `Reg::RegressionBase`, which standardizes the training, prediction and diagnostic workflow regardless of the model chosen. This design makes it straightforward to swap models, plug them into the validation pipeline or compare their results under identical conditions.

Every model exposes the same entry points: `fit()` to train and compute diagnostics, `predict()` to generate predictions on new data and `summary()` to display a structured report of the estimated coefficients and model statistics. Under the hood, each model handles its own estimation logic while delegating the statistical inference to the shared diagnostic toolkit described in [**IV - Statistical Functions**](/docs/IV_stats.md).

```cpp
// Common interface shared by all models
model.fit(X, y);                                    // train + compute diagnostics
std::vector<double> y_pred = model.predict(X_new);  // predict
model.summary();                                    // display results
model.summary(true);                                // detailed output
```

Each fitted model can also shows its results through a set of getters:

```cpp
model.get_coeffs();             // estimated β coefficients
model.get_intercept();          // intercept β₀
model.get_stats();              // general model statistics (R², F-stat, ...)
model.get_coefficient_stats();  // per-coefficient stats (SE, t-stat, p-value)
```

All Regression models can be imported at once through a single include: 
- `#include "Models/Supervised/Regressions.hpp"`

## Linear Regression

Linear regression is the most fundamental estimation method and the natural starting point for any regression task. It estimates the coefficients `β` by minimizing the sum of squared residuals, yielding the **OLS** estimator `β = (XᵀX)⁻¹Xᵀy`. When the error variance structure is known, the model can instead be estimated via **GLS**, which incorporates a covariance matrix `Ω` to produce more efficient estimates under heteroskedasticity or autocorrelation.

A key design decision when using linear regression is the choice of covariance estimator, which directly determines the reliability of standard errors, t-statistics and confidence intervals. The available options are:

- **`"classical"`** (default): standard OLS covariance, valid under homoskedasticity and no autocorrelation
- **`"HC3"`**: heteroskedasticity-consistent estimator, robust to non-constant error variance
- **`"HAC"`**: Newey-West estimator, robust to both heteroskedasticity and autocorrelation, typically used for time series
- **`"cluster"`**: clusters standard errors within groups, suitable when observations share within-group correlation
- **`"GLS"`**: incorporates a known variance structure `Ω` for generalized least squares estimation

```cpp
// OLS with classical standard errors (default)
Reg::LinearRegression model;

// OLS with HC3 robust standard errors
Reg::LinearRegression model("HC3");

// Clustered standard errors
Reg::LinearRegression model("cluster", cluster_ids);

// GLS with known variance structure
Reg::LinearRegression model("GLS", {}, std::move(Omega));

model.fit(X, y);
model.summary();
```

**Decision flow**: Start with `"classical"` and use the Breusch-Pagan and Durbin-Watson diagnostics (see [**IV**](/docs/IV_stats.md)) to check whether the OLS assumptions hold. Switch to `"HC3"` if heteroskedasticity is detected, to `"HAC"` if autocorrelation is present, or to `"cluster"` when observations are grouped. Use `"GLS"` only when the variance structure `Ω` is explicitly known.

## Regularized Regressions

 When predictors are highly correlated or simply when the number of features is large relative to the number of observations, OLS estimates tend to become unstable and overfit the training data. Regularized regression methods address this by adding a penalty term to the objective function that shrinks the coefficients toward zero.

All three regularized models share an `optimal_lambda()` utility (where applicable) that automates hyperparameter selection via `Validation::GSearchCV` over a log-spaced grid.

### Ridge Regression

Ridge regression adds an **L2 penalty** to the OLS objective, yielding a closed-form solution `β = (XᵀX + λI)⁻¹Xᵀy`. Ridge is particularly well-suited when predictors are correlated and shrinks all coefficients toward zero but never sets any of them exactly to zero, so it retains all features in the model.

```cpp
Reg::RidgeRegression model(1.0);   // lambda = 1.0 (default)
model.fit(X, y);
model.summary();

// Automated lambda selection via GridSearchCV over a log-spaced grid
model.optimal_lambda(1e-4, 1e2, 50, X, y);  // search in [1e-4, 1e2] with 50 steps
```

**Note**: Ridge is the choice when multicollinearity is present and all predictors are expected to contribute to the outcome (the effective degrees of freedom decrease as `λ` increases, reflecting the growing shrinkage applied to the coefficients). A larger `λ` produces stronger shrinkage and a simpler model; use `optimal_lambda()` or cross-validation to find the right balance.

### Lasso Regression

Lasso regression adds an **L1 penalty** to the objective. Unlike Ridge, the L1 penalty induces sparsity: as λ increases, some coefficients are shrunk exactly to zero, effectively performing automatic feature selection. This makes Lasso especially useful in high-dimensional settings where only a subset of predictors is expected to be relevant.

```cpp
Reg::LassoRegression model(0.1);   // lambda = 0.1 (default)
model.fit(X, y); 
model.summary();

// Automated lambda selection
model.optimal_lambda(1e-4, 1e1, 50, X, y);
```

**Note**: Prefer Lasso over Ridge when you suspect that only a few predictors are truly relevant and interpretability matters. Be cautious with highly correlated features: Lasso tends to arbitrarily select one and discard the others, which is where Elastic Net becomes more appropriate.

### Elastic Net Regression

Elastic Net combines both the **L1 and L2 penalties**. This hybrid formulation inherits the sparsity-inducing property of Lasso while retaining the stability of Ridge in the presence of correlated predictors. The `l1_ratio` parameter controls the balance between the two penalties: a value of `1.0` for a pure Lasso model, while `0.0` is for a pure Ridge model.

```cpp
Reg::ElasticRegression model(0.1, 0.5);  // alpha = 0.1, l1_ratio = 0.5 (default)
model.fit(X, y);
model.summary();

double df = model.effective_df(X_c);
```

**Note**: Elastic Net is generally the most robust regularized option and a safe default when the relationship between predictors is unknown. Use a `l1_ratio` closer to `1.0` to favor sparsity, or closer to `0.0` to favor stability under collinearity. Tune both `alpha` and `l1_ratio` jointly using `Validation::RSearchCV` for efficiency.

## Stepwise Regression

Stepwise regression takes a different approach to model selection: rather than shrinking coefficients, it explicitly selects or removes predictors from the model based on a statistical criterion evaluated at each step.

Three search strategies are available:
- **`"forward"`**: starts from an empty model and adds the most significant predictor at each step
- **`"backward"`** (default): starts from the full model and removes the least significant predictor at each step
- **`"stepwise"`**: combines both directions, re-evaluating all predictors at every step

The entry and exit of predictors is governed by a threshold criterion, either a significance level `alpha` or an information criterion:
- **`"alpha"`** (default): predictors enter if their p-value falls below `alpha_in` and exit if it exceeds `alpha_out`
- **`"aic"`** / **`"bic"`**: predictors are selected to minimize the Akaike or Bayesian Information Criterion, which penalizes model complexity and thus tend to produce more parsimonious models than pure significance-based selection

```cpp
// Backward stepwise with significance threshold (default)
Reg::StepwiseRegression model({0.05, 0.10}, "backward", "alpha");

// Forward stepwise with BIC criterion
Reg::StepwiseRegression model({}, "forward", "bic");

model.fit(X, y);
model.summary();
```

**Note**: Backward elimination is generally preferred over forward selection as it starts from the full model and is less likely to miss important interaction effects. Use `"aic"` or `"bic"` as the threshold when the goal is model parsimony rather than strict significance control. For multicollinearity prefer Ridge or Elastic Net models.

## Choosing a Model

| Scenario | Recommended model |
|---|---|
| Standard linear inference, well-behaved data | `LinearRegression("classical")` |
| Heteroskedastic errors | `LinearRegression("HC3")` |
| Autocorrelated errors / time series | `LinearRegression("HAC")` |
| Grouped / panel data | `LinearRegression("cluster")` |
| Known variance structure | `LinearRegression("GLS")` |
| Multicollinearity, all features relevant | `RidgeRegression` |
| High-dimensional, sparse signal expected | `LassoRegression` |
| Correlated features + sparsity | `ElasticRegression` |
| Exploratory feature selection | `StepwiseRegression` |

**To test it yourself**, you can also check the corresponding files:
- [**Regressions hpp**](/include/Models/Supervised/Regression)
- [**Regressions cpp**](/src/Models/Supervised/Regression)
- [**Test folder**](/tests/)

To read the next part: [**VIII**].