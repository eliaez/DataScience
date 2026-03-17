# V - Preprocessing

Raw data is rarely ready to be fed directly into a model. That's why we need Preprocessing, it refers to the set of transformations applied to the data before training in order to improve numerical stability, accelerate convergence, handle missing values and ensure that the model assumptions are better satisfied. Thus, this part is organized around three steps: Imputation, Scaling, Distribution Transformation and Dimensionality Reduction.

All preprocessing utilities (`Imputation`, `Scaling`, `Split`, `PCA`) can be imported at once via a single include: `#include "Preprocessing/Preprocessing.hpp"`

## Imputation

Datasets frequently contain missing values (NaN), which most models cannot handle natively. To address this, Imputation fills these gaps using statistical estimates derived from the available data, preserving the dataset size while minimizing the distortion introduced:

- **`"mean"`** (default): replaces by the column mean, suitable for symmetric distributions without extreme outliers

- **`"median"`**: replaces by the median, robust to outliers and preferred for skewed distributions

- **`"mode"`**: replaces by the most frequent value, suitable for categorical-like columns

- **`"forward"`** / **`"backward"`**: propagates the previous/next valid value, suitable for time series where temporal continuity matters

```cpp
// Either by column index or column name
Imputation::imputation(X, 2, "median");  
Imputation::imputation(X, "income", "mean");
Imputation::imputation(X, "date_col", "forward");
```
<br>

- **`KNN Imputation`**: for more sophisticated imputation, it will estimates missing values by looking at the `K` nearest neighbours in the dataset and averaging their values, weighted by proximity (`KNN` approach captures local structure in the data such as correlated features):

```cpp
// Standard KNN imputation with Euclidean distance Lp_norm = 2 (continuous data type)
Imputation::KNN_imputer(X, 5);

// KNN with custom Lp norm
Imputation::KNN_imputer(X, 5, 3);

// KNN with Gower distance (mixed data types)
Imputation::KNN_imputer(X, 5, -1);
```

**Note:** Imputation is reasonable up to roughly 10-15% missing data; beyond that the introduced bias may significantly distort the column's distribution.

## Scaling

Models might be sensitives to the scale of input features. That's why we use Scaling to ensure that all features contribute on equal footing during training. The following methods are available:

- **`"standard"`** (default): z-score normalization `(x − μ) / σ`, produces zero mean and unit variance, the most common choice for linear models and regularization

- **`"mean"`**: centers the column `(x − μ) / (max − min)`, preserves the shape while centering around zero

- **`"minmax"`**: rescales to `[min, max]` (default `[0, 1]`), useful when bounded inputs are required

- **`"percentile"`**: rescales based on percentile rank, robust to outliers

```cpp
// Either by column index or column name
Scaling::scaling(X, 0, "standard"); // z-score
Scaling::scaling(X, "price", "minmax", 0.0, 1.0);  // rescale to [0,1]
```

## Distribution Transformation

Even after scaling, a heavily skewed distribution may not follow the normality assumptions underlying many models. Distribution Transformations modify the shape of the distribution itself, making it more symmetric and closer to normality before any scaling is applied (**Need to be applied before scaling**). The following transformations are available:

- **`"log"`** (default): `log(x)`, suitable for right-skewed positive data such as prices or counts

- **`"box_cox"`**: parametric power transform, requires strictly positive values

- **`"yeo_johnson"`**: extends Box-Cox to zero and negative values

- **`"power"`**: applies `x^λ`, useful when a specific power transform is known in advance

```cpp
// Either by column index or column name
Scaling::transform(X, 1, "log");
Scaling::transform(X, 1, "box_cox",  0.5);     // with λ
Scaling::transform(X, "temp",  "yeo_johnson");
```

## Train/Test Split

Before any training or evaluation, the dataset must be divided into non-overlapping subsets to ensure that model performance is measured on data it has never seen. Then, splits `X` and `y` into train/test (or train/validation/test) sets by using one of those two strategies available:

- **`train_test_split`**: randomly assigns observations to each subset according to the specified proportions, suitable for most use cases

- **`stratified_split`**: preserves the class distribution of `y` in each subset, recommended for imbalanced datasets

```cpp
// 80% train / 20% test
Split::TrainTestSplit s = Split::train_test_split(X, y, 80, /*shuffle=*/true);
s.X_train; s.X_test; s.y_train; s.y_test;

// 60% train / 20% valid / 20% test (stratified)
Split::TrainTestValidSplit sv = Split::stratified_split(X, y, {60, 20}, true);
sv.X_train; sv.X_valid; sv.X_test;
sv.y_train; sv.y_valid; sv.y_test;
```
**Note**: use shuffle unless working with time series data, the stratified splits when classes are imbalanced to avoid folds with missing classes and the three-way split (train/valid/test) when tuning hyperparameters.

## Dimensionality Reduction - PCA

Once the data is cleaned, scaled and split, it may still suffer from the **curse of dimensionality** (too many features slow down training, increase overfitting risk and make visualisation impossible). 

That's why, we can solve this by using **Principal Component Analysis (`PCA`)** method that identifies the directions of maximum variance in the data (called principal components) and projects the observations onto the top `k` of them. Let's see our examples:
```cpp
// Automatic k selection via cumulative explained variance
PCA pca;
pca.fit(X_train); // Use scaled data 
Dataframe X_reduced = pca.transform(X_test);

// Or in one step
Dataframe X_reduced = pca.fit_transform(X_train);

// Manual k selection
PCA pca(3);
Dataframe X_reduced = pca.fit_transform(X_train);

// Inspect results
pca.get_nb_components();       // k actually used
pca.get_eigen_values();        // eigenvalues
pca.get_explained_variance();  // variance ratio per component
pca.get_eigen_vectors();       // principal axes (col-major)
```
**Note**: **`SVD`** was not implemented from scratch unlike to all others algorithms and linear algebra operations due to the complexity of the numerous steps: Bidiagonalization via Householder reflections, followed by an iterative QR algorithm with shifts (Golub-Kahan) to converge to diagonal form, all while carefully handling convergence criteria, deflation strategies, and edge cases (zero singular values,... ). Moreover, each of these steps is a potential source of floating-point instability that can silently corrupt results.

**To test it yourself**, you can also check the corresponding folders:
- [**Preprocessing hpp**](/include/Preprocessing)
- [**Preprocessing cpp**](/src/Preprocessing)

To read the next part: [**VI - Validation**](/docs/VI_validation.md).