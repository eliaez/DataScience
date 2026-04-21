# IX - Classifications

Classification models follow the same design philosophy as the regression module: all models inherit from `Class::ClassificationBase`, which standardises the training, prediction and diagnostic workflow regardless of the algorithm chosen. This common interface makes it straightforward to swap models, plug them into the validation pipeline or compare their performance under identical conditions.

Every model exposes the same entry points: `fit()` to train and compute diagnostics (use only `fit_without_stats()` to train then `compute_stats()`,...), `predict()` to output class labels on new data, `predict_proba()` to return probability estimates and `summary()` to display a structured performance report. Under the hood, each model handles its own estimation logic while delegating the statistical inference to the shared diagnostic toolkit described in [**IV - Statistical Functions**](/docs/IV_stats.md).

```cpp
// Common interface shared by all models
model.fit(X, y);                                          // train + compute diagnostics
model.summary();                                          // display results
model.summary(true);                                      // detailed output
std::vector<double> y_pred  = model.predict(X_new);       // predict class labels
std::vector<double> y_proba = model.predict_proba(X_new); // predict probabilities

// Or, for split train/test workflows:
model.fit_without_stats(X_train, y_train);
model.compute_stats(X_test, x_const, y_test);
model.summary(true);
```

The base class further offers a small set of setters to configure shared training parameters before fitting, notably `set_tol()`, `set_maxiter()`, `set_learningrate()` and `set_refclass()` for the reference category in multinomial settings.

All classification models can be imported at once through a single include:
- `#include "Models/Supervised/Classifications.hpp"`

## Logistic Regression

Logistic regression is the natural starting point for classification tasks. Logistic regression models log-odds as a linear combination of predictors and estimates coefficients by maximizing log-likelihood via gradient descent. In the multiclass case the model automatically switches to a **softmax** formulation, treating each class through a one-vs-rest decomposition. 

Moreover, you have the choice of regularisation which controls overfitting and the stability of the coefficient estimates:

- **`"l2"`** (default): penalises the sum of squared coefficients, it shrinks all coefficients but never eliminates any of them.
- **`"l1"`**: penalises the sum of absolute coefficients, driving irrelevant features coefficients to zero (exactly).
- **`"elasticnet"`**: combines both penalties through an `l1_ratio` parameter, it can be controlled by the `l1_ratio` parameter (`1.0` is pure L1).
- **`""`**: no regularisation, suitable only when the number of observations is large relative to the number of features and multicollinearity is not a concern.

The regularisation strength is governed by `C = 1 / λ`; a smaller `C` corresponds to stronger regularisation. When the optimal value is unknown, `optimal_c()` automates the search via `Validation::GridSearchCV` over a log-spaced grid.

```cpp
// L2 regularisation (default)
Class::LogisticRegression model;

// L1 regularisation with a specific C
Class::LogisticRegression model(0.5, "l1");

// Elastic Net with l1_ratio = 0.7
Class::LogisticRegression model(1.0, "elasticnet", 0.7);

model.fit(X, y);
model.summary();

// Automated C selection over [1e-3, 1e3] with 50 steps
model.optimal_c(1e-3, 1e3, 50, X, y);
```

## Support Vector Machine

The SVM finds the hyperplane that maximises the margin between classes. Only a subset of points (the **support vectors**) actually determine the boundary. The `C` parameter controls the trade-off between margin width and training misclassification: a large `C` enforces a narrow, low-error margin while a small `C` allows more violations in exchange for a smoother boundary.

The expressive power of the model is extended through the **kernel**:

- **`"linear"`** (default): no feature mapping; fast and interpretable, best suited when classes are linearly separable in the original space.
- **`"rbf"`**: Radial Basis Function kernel measuring Gaussian similarity between points. It's the most versatile choice and often the best starting point for non-linear problems.
- **`"poly"`**: polynomial kernel of degree `degree`, it captures structured interactions between features.

The `gamma` parameter applicable to `"rbf"` and `"poly"`, controls how far the influence of each training point extends. `"scale"` (default) sets `γ = 1 / (p · Var(X))` and adapts to the feature scale while `"auto"` uses `γ = 1 / p`.

```cpp
// Linear SVM (default)
Class::SVM_Algo model;

// RBF kernel with C = 10.0 and gamma = "scale"
Class::SVM_Algo model(10.0, "rbf", "scale");

// Polynomial kernel of degree 3
Class::SVM_Algo model(1.0, "poly", "auto", 3);

model.fit(X, y);
model.summary();
```

**Note**: SVMs are sensitive to feature scaling; always standardise your inputs before fitting. Use `"linear"` when interpretability matters or the dataset is high-dimensional and sparse.

## Random Forest

Random Forest builds an ensemble of decision trees, each trained on a bootstrap sample of the data and a random subset of features at every split. The final prediction aggregates the individual trees through **majority vote**, which reduces variance without meaningfully increasing bias. The degree of randomness injected at the feature level is controlled by `max_features`:

- **`"sqrt"`** (default): selects `√p` features at each split.
- **`"log2"`**: selects `log₂(p)` features, producing more diverse trees and often better results on very high-dimensional data.
- **`"all"`**: considers all features at every split.
- A float string such as `"0.5"` selects a fixed fraction of features.

```cpp
// Default Random Forest
Class::RandomForest model;

// Random Forest model
Class::RandomForest model(
    200,       // Nb of tree
    10,        // Max depth of a tree (unlimited by default)
    5,         // min_samples_split (2 by default)
    2,         // min_samples_leaf  (1 by default)
    "sqrt",    // max_features
    "entropy"  // "entropy" or "gini" (default): criterion to evaluate candidate splits
);

Dataframe features_imp = model.fit_without_stats(X, y);
model.compute_stats(X_test, features_imp, y_test);
model.summary(true);
```

**Note**: Random Forests are robust to outliers and require little preprocessing (no scaling is needed).

## XGBoost

XGBoost constructs the trees **sequentially**, with each new one fitting the residual gradients of the previous ensemble. This second-order boosting procedure optimises a regularised objective at every step, combining a loss function with L1 and L2 penalties on the leaf weights to control model complexity.

- **`lambda`** (L2, default `1`): shrinking leaf weights toward zero.
- **`alpha`** (L1, default `0`): pruning low-contribution leaves.
- **`gamma`**: minimum loss reduction required to allow a further split. A non-zero value acts as a conservative pruning criterion, preventing splits that do not yield a meaningful gain.

```cpp
// Default XGBoost
Class::XGBoost model;

// XGBoost model
Class::XGBoost model(
    0.1,        // gamma
    0.5,        // alpha L1
    1.0,        // lambda L2
    150,        // Nb of tree
    6,          // Max depth of a tree (unlimited by default) 
    3           // min_child_weight (1 by default)
);

Dataframe features_imp = model.fit_without_stats(X, y);
model.compute_stats(X_test, features_imp, y_test);
model.summary(true);

// Or access raw probabilities
std::vector<double> proba = model.predict_proba(X_new);
```

## Neural Network

The Neural Network provides a flexible multi-layer architecture where both the layer sizes and activation functions are user-defined. Each `Layer` performs a standard affine transformation followed by an activation, trained via **backpropagation** with the **Adam** optimiser, mini-batch gradient descent and the optionals functions: **L2 regularisation** and **dropout**.

Let's take a look at the available activations:

- **`RELU`**: the standard choice for hidden layers.
- **`SOFTMAX`**: should be placed on the output layer for multiclass classification, producing a probability distribution over classes.
- **`TANH`**: maps inputs to (−1, 1); smooth and zero-centred, making it preferable to ReLU in shallow networks or when features are centred.
- **`LINEAR`**: returns the input unchanged; useful as the output activation in regression heads embedded in a classification pipeline.

```cpp
// Two hidden layers (64, 32) with ReLU, softmax output
Class::NeuralNetwork model(
    {64, 32, nb_classes},
    {ActivationType::RELU, ActivationType::RELU, ActivationType::SOFTMAX},
    100,   // n_epochs
    0.01,  // lambda (L2 weight decay)
    0.9,   // beta1 (Adam moment decay rates)
    0.999, // beta2 (Adam moment decay rates)
    0.8,   // keep_prob sets the dropout retention probability (set to `1.0` to disable)
    32     // batch_size
);

Dataframe loss_history = model.fit_without_stats(X_train, y_train);
model.compute_stats(X_test, loss_history, y_test);
model.summary(true);
```

**Note**: Start with a simple architecture (one or two hidden layers) and increase capacity (only if underfitting is observed).

## Choosing a Model

| Scenario | Recommended model |
|---|---|
| Baseline, interpretable classification | `LogisticRegression("l2")` |
| Sparse signal, feature selection needed | `LogisticRegression("l1")` |
| Correlated features + sparsity | `LogisticRegression("elasticnet")` |
| Non-linear boundary, robust baseline | `RandomForest` |
| Strong predictive performance, structured data | `XGBoost` |
| Non-linear boundary, well-scaled features | `SVM_Algo("rbf")` |
| High-dimensional, linearly separable | `SVM_Algo("linear")` |
| Complex patterns, large datasets | `NeuralNetwork` |

**To test it yourself**, you can also check the corresponding files:
- [**Classifications hpp**](/include/Models/Supervised/Classification)
- [**Classifications cpp**](/src/Models/Supervised/Classification)
- [**Test folder**](/tests/)