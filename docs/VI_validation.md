# VI - Validation

Toolkit for model evaluation and hyperparameter tuning. Before deploying a model, it's essential to assess how well it generalizes to unseen data and to identify the optimal set of hyperparameters. Simply evaluating a model on its training data leads to overfitting and overly optimistic performance estimates.

All validation utilities operate on any model inheriting from `Reg::RegressionBase` and support three metrics: `"mse"` (default), `"mae"` and `"r2"`.

## Cross-Validation

Cross-validation is the standard approach to estimate the true generalization performance of a model. The method will train `k` times the model, each time using `k-1` folds for training and the remaining fold for evaluation. `Cross_validation` returns per-fold scores, their mean and standard deviation across folds.

```cpp
Validation::CVres cv = Validation::cross_validation(
    &model, X, y,
    /*k=*/5,
    /*metric=*/"mse",
    /*shuffle=*/true,
    /*show_progression=*/true
);

cv.scores;      // score for each fold
cv.mean_score;  // mean across folds
cv.std_score;   // std across folds
```

**Note**: A low `std_score` relative to `mean_score` indicates stable generalization. Moreover, high variance between folds suggests overfitting or insufficient data.

## Grid Search CV

Once cross-validation is set up, the next step is finding the best hyperparameters.
Grid search will evaluate every possible combination of hyperparameters via cross-validation. `param_grid` must list all constructor parameters **in order**, with a single-element sub-vector for fixed parameters.

```cpp
// Example: fixed param="l1", search over {1,2,3,4,5}, fixed param=2
std::vector<std::vector<std::variant<double, std::string>>> grid = {
    {"l1"}, 
    {1, 2, 3, 4, 5}, 
    {2}
};

Validation::GSres gs = Validation::GSearchCV(
    &model, X, y, grid,
    /*k=*/5,
    /*metric=*/"mse",
    /*shuffle=*/true
);

gs.best_score;    // best CV score found
gs.best_params;   // corresponding parameters
gs.all_results;   // full history: vector of (params, score)
```

## Random Search CV

Samples hyperparameter combinations randomly over defined ranges, more efficient than grid search in high-dimensional spaces. 

Each parameter range is defined as `{[min, max], log_scale}`: use a single-element vector for fixed parameters, `log=true` for log-uniform sampling (suitable for learning rates, regularization), `log=false` for uniform sampling.

```cpp
// Example: fixed param=0, log-uniform in [1,100], uniform in [2,5], uniform in ["l1", "l2", "elasticnet"] (bool haven't any influence on string params)
std::vector<std::pair<std::vector<double>, bool>> ranges = {
    {{0},     false},
    {{1, 100}, true},
    {{2, 5},  false},
    {{"l1", "l2", "elasticnet"}, true}
};

Validation::GSres rs = Validation::RSearchCV(
    &model, X, y, ranges,
    /*k=*/5,
    /*metric=*/"mse",
    /*nb_iter=*/50,
    /*shuffle=*/true
);

rs.best_score;   // best CV score found
rs.best_params;  // corresponding parameters
rs.all_results;  // full history: vector of (params, score)
```

**Note**: Prefer `GSearchCV` for small grids with few parameters, `RSearchCV` for larger search spaces where exhaustive search is too costly. Furthermore, increase `nb_iter` to trade computation time for better coverage.

**To test it yourself**, you can also check the corresponding files:
- [**Validation.hpp**](/include/Validation/Validation.hpp)
- [**Validation.cpp**](/src/Validation/Validation.cpp)

To read the next part: [**VII - Regressions**](/docs/VII_regressions.md).