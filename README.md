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

**To test it yourself**, you can check the corresponding [**Test file**](/tests/backend/tests_data.cpp) or **the following document** to have an idea of how to use the various functions. For further details, see [**I - Dataframe**](/docs/I_dataframe.md).

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

**To test it yourself**, you can check the [**Tests folder**](/tests/) or **the following document** to have an idea of how to use the various backends and their functions. For further details, see [**II - Linear Algebra**](/docs/II_linalg.md).

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

Once the model is estimated, we need tools to assess its reliability and quality. This section provides a statistical toolkit covering both basic descriptive statistics (`mean`, `var`, `cov`, ...) and advanced diagnostic tests, all automatically called during the regression pipeline but also usable as standalone utilities:

- **Covariance matrix** with multiple estimators (`classical`, `HC3`, `HAC`, `cluster`, `GLS`)
- **Goodness-of-fit metrics** including R², MAE, MSE and RMSE
- **Hypothesis testing** through Fisher, Student, Durbin-Watson and Breusch-Pagan tests
- **Multicollinearity and model selection** via VIF and AIC/BIC
- **Time series analysis** through ARIMA/SARIMA automatic parameter detection via stationarity testing, seasonality detection and autocorrelation analysis

For further details, see [**IV - Statistical Functions**](/docs/IV_stats.md).

## V - Preprocessing

Before training a model, raw data rarely comes in a form suitable for direct use: features may live on incompatible scales, distributions can be heavily skewed, missing values corrupt otherwise clean datasets and the sheer number of features can slow down training or lead to overfitting. The preprocessing utilities address these concerns by providing column-wise scaling, distribution transforms, imputation strategies and dimensionality reduction::

- **Scaling** the data with different methods (`z-score`, `mean-centering`, `min-max`, `percentile`)
- **Distribution transformation** (`log`, `Box-Cox`, `Yeo-Johnson` and `power`)
- **Imputation** (`mean`, `median`, `mode`, `forward`/`backward` or `KNN`)
- **Train/Test split** methods with random and stratified splits options
- **Dimensionality reduction** via PCA with automatic or manual component selection

For further details, see [**V - Preprocessing**](/docs/V_preprocessing.md).

## VI - Validation

Furthermore, once the model is trained, evaluating it requires more than a single train/test split. The validation toolkit provides robust evaluation protocols through cross-validation and Grid Search/Random Search, ensuring that reported performance reflects true generalization rather than overfitting to a particular data partition:

- **Cross-validation** through k-fold evaluation
- **Grid Search & Random Search CV** for hyperparameter tuning and are both relying on cross-validation

For further details, see [**VI - Validation**](/docs/VI_validdation.md).

## VII - Regressions

With the former sections, we now have everything needed to build and evaluate regression models. This section brings it all together by providing a unified regression framework covering a wide range of estimation strategies, from standard OLS to regularized and stepwise approaches.

All models share a common interface through `Reg::RegressionBase`, which standardizes training, prediction and diagnostic reporting regardless of the model chosen. The following models are currently available:

- **Linear Regression** with multiple covariance estimators (`classical`, `HC3`, `HAC`, `cluster`, `GLS`)
- **Ridge Regression** for L2 regularization
- **Lasso Regression** for L1 regularization
- **Elastic Net** combining both L1 and L2 penalties
- **Stepwise Regression** for automated feature selection through forward, backward or bidirectional search strategies

**To test it yourself**, you can check the corresponding [**Test folder**](/tests/regression) or **the following document** to have an idea of how to use the various functions, see [**VII - Regressions**](/docs/VII_regressions.md).

## VIII - 

**In progress...**
