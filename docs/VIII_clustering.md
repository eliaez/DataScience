# VIII - Clustering

Toolkit for unsupervised clustering. Unlike regression or classification, clustering aims to discover natural groupings in data without labels. The main algorithm provided is **K-Means** which partitions data into `k` clusters by minimizing intra-cluster variance (inertia).

## K-Means

K-Means is the standard approach for partitioning data into a fixed number of clusters. The algorithm is initialized using **K-Means++** to ensure better starting centroids and more stable results. Each run minimizes inertia, and the best result over `n_init` runs is kept.

```cpp
// k=-1 lets the algorithm select the optimal number of clusters automatically
Kmeans model(
    /*k=*/3,
    /*n_init=*/10,
    /*max_iter=*/300,
    /*method=*/"kmeans"
);

std::vector<int> labels = model.fit_predict(df, /*show_progression=*/true);

model.get_inertia();          // final inertia (sum of squared distances to centroid)
model.get_cluster_centers();  // centroid coordinates per cluster
```

### Automatic cluster selection

If you don't know how many clusters to use, simply set `k=-1`. The algorithm will then automatically select the optimal number of clusters using the **elbow method**, searching over a defined range.

```cpp
Kmeans model(
    /*k=*/-1,
    /*n_init=*/10,
    /*max_iter=*/300,
    /*method=*/"kmeans",
    /*min_max=*/{2, 10}   // search between 2 and 10 clusters
);
```

## Variants

Three algorithmic variants are available via the `method` parameter, each offering a different trade-off between accuracy and speed:

- **`"kmeans"`** *(default)*: Batch K-Means — centroids are updated after processing all `N` points. Most accurate, recommended for small to medium datasets.
- **`"minibatch"`**: Mini-Batch K-Means — centroids are updated after processing `sqrt(N)` points. Significantly faster on large datasets with a minor accuracy trade-off.
- **`"online"`**: Online K-Means — centroids are updated after each individual point. Suitable for streaming or very large datasets.

## Predict on new data

Once the model is fitted, you can assign cluster labels to new data points without retraining using `predict`.

```cpp
std::vector<int> new_labels = model.predict(new_df);
```

## Key parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `k` | Number of clusters (`-1` for automatic selection) | `-1` |
| `n_init` | Number of independent runs, best result is kept | `10` |
| `max_iter` | Maximum iterations per run | `300` |
| `method` | Algorithm variant: `"kmeans"`, `"minibatch"`, `"online"` | `"kmeans"` |
| `min_max` | Search range for automatic `k` selection | `{2, 10}` |

**To test it yourself**, you can also check the corresponding files:
- [**Kmeans.hpp**](/include/Models/Unsupervised/Kmeans.hpp)
- [**Kmeans.cpp**](/src/Models/Unsupervised/Kmeans.cpp)

To read the next part: [**IX**].