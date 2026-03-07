# Machine Learning From Scratch
Educational project implementing fundamental machine learning and regression algorithms in C++ **without relying on external ML libraries**.

## Prerequisites

- **C++20** compiler (MSVC recommended, MinGW-GCC supported)
- **CMake** ≥ 3.15
- **OpenMP**
- **Eigen 3.4.1** *(auto-fetched via FetchContent)*
- **Google Benchmark** *(auto-fetched via FetchContent)*
- **Boost 1.90.0** — set path in CMakeLists.txt: `C:/Program Files (x86)/Boost/boost_1_90_0`
- **Intel MKL + TBB** *(optional, MSVC only)*
- **AVX2** *(optional, auto-detected)*

## I - Dataframe
As a first step, we need a class to handle and manipulate our data. Thus, our **Dataframe** class which provides the structure for numerical data manipulation and is specifically designed for machine learning and linear algebra operations. It supports both row-major and column-major storage. Additionally, the class handles heterogeneous data types through automatic categorical encoding.

**To test it yourself**, you can check the corresponding [**Test file**](/tests/backend/tests_data.cpp) or **the following document** to have an idea of how to use the various functions. <br>
For further details, see [**I - Dataframe**](/docs/I_dataframe.md).

## II - Linear Algebra
Before implementing any ML and regression algorithms, we need fundamental linear algebra operations and functions. It's the core of the optimization process to get better performance, so we will implement 3 different approaches and add 2 external libraries to compare:

- Implemented:
  - Naive
  - AVX2
  - AVX2 Threaded
- External:
  - Eigen
  - MKL

Moreover, to enable backend selection and operation dispatching, the **`Linalg`** namespace was created to provide a unified interface. This abstraction layer enables transparent backend switching without altering function signatures.

**To test it yourself**, you can check the [**Tests folder**](/tests/) or **the following document** to have an idea of how to use the various backends and their functions. <br>
For further details, see [**II - Linear Algebra**](/docs/II_linalg.md).

## III - Linear Algebra Benchmark
Comparison between backends using Google Benchmark across 4 operations:


| Operation           | Backend Ranking                     |
|:--------------------|:-----------------------------------:|
| Transpose           | Naive < Eigen < AVX2TH < AVX2 = MKL |
| In-place Transpose  | Naive < Eigen < AVX2 < MKL < AVX2TH |
| Matrix Multiply     | Naive < AVX2 < AVX2TH < Eigen = MKL |
| Inverse             | Naive < AVX2 < AVX2TH < Eigen = MKL |

For further details, see [**III - Linear Algebra Benchmark**](/docs/III_benchmark.md).

## IV - Statistical Functions

Once the model is estimated, we need tools to assess its reliability and quality. Indeed, a model can fit the data well on paper while still violating key statistical assumptions leading to biased inference or unreliable predictions. This section provides a statistical toolkit covering both basic descriptive statistics (`mean`, `var`, `cov`, ...) and advanced diagnostic tests, all automatically called during the regression pipeline but also usable as standalone utilities:

- **Covariance matrix** with multiple estimators (`classical`, `HC3`, `HAC`, `cluster`, `GLS`) to handle heteroskedasticity, autocorrelation and clustered data
- **Goodness-of-fit metrics**: R², adjusted R², MAE, MSE and RMSE to evaluate model performance and prediction error
- **Hypothesis testing** through Fisher and Student tests for global and individual significance, Durbin-Watson and Breusch-Pagan to detect autocorrelation and heteroskedasticity in residuals
- **Multicollinearity detection** via VIF to identify correlated predictors
- **Model selection criteria** through AIC and BIC penalizing model complexity via effective degrees of freedom, used for regularized models

**To test it yourself**, you can check the **the following document** to have an idea of how to use the various functions. <br>
For further details, see [**IV - Statistical Functions**](/docs/IV_stats.md).

## V - Preprocessing & Validation

Before training a model, raw data rarely comes in a form suitable for direct use: features may live on incompatible scales, distributions can be heavily skewed and missing values corrupt otherwise clean datasets. The preprocessing utilities address these concerns by providing column-wise scaling, distribution transforms and imputation strategies:

- **Scaling** the data with different methods (`z-score`, `mean-centering`, `min-max`, `percentile`) to bring features onto comparable scales before training
- **Distribution transformation** like `log`, `Box-Cox`, `Yeo-Johnson` and `power` to transform and correct skewed distributions prior to scaling
- **Imputation** methods on columns will handle missing values with a `mean`, `median`, `mode` or `forward`/`backward` fill
- **Train/Test split** methods with random and stratified splits options, to split into two or three subsets (train / validation / test)

Furthermore, once the model is trained, evaluating it honestly requires more than a single train/test split. The validation toolkit provides robust evaluation protocols through cross-validation and automated hyperparameter search, ensuring that reported performance reflects true generalization rather than overfitting to a particular data partition:

- **Cross-validation** through k-fold evaluation will return per-fold scores, mean and standard deviation for stable generalization estimates
- **Grid Search & Random Search CV** are automated hyperparameter tuning over discrete grids or continuous ranges and are both relying on cross-validation to select the best configuration

**To test it yourself**, you can check the **the following document** to have an idea of how to use the various functions. <br>
For further details, see [**V - Preprocessing & Validation**](/docs/V_preprocess_valid.md).

## VII - Regressions

**In progress...**