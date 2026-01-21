# Machine Learning From Scratch
Educational project implementing fundamental machine learning and regression algorithms in C++ **without relying on external ML libraries**.

## I - Dataframe
As a first step, we need a class to handle and manipulate our data. Thus, our **Dataframe** class which provides the structure for numerical data manipulation and is specifically designed for machine learning and linear algebra operations. It supports both row-major and column-major storage. Additionally, the class handles heterogeneous data types through automatic categorical encoding.

**To test it yourself**, you can check the corresponding [**Test file**](tests/tests_data.cpp) or **the following document** to have an idea of how to use the various functions. <br>
For further details, see [**I - Dataframe**](docs/I_dataframe.md).

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

**To test it yourself**, you can check the [**Tests folder**](tests/) or **the following document** to have an idea of how to use the various backends and their functions. <br>
For further details, see [**II - Linear Algebra**](docs/II_linalg.md).

## III - Linear Algebra Benchmark
Comparison between backends using Google Benchmark across 4 operations:


| Operation           | Backend Ranking                     |
|:--------------------|:-----------------------------------:|
| Transpose           | Naive < Eigen < AVX2TH < AVX2 = MKL |
| In-place Transpose  | Naive < Eigen < AVX2 < MKL < AVX2TH |
| Matrix Multiply     | Naive < AVX2 < AVX2TH < Eigen = MKL |
| Inverse             | Naive < AVX2 < AVX2TH < Eigen = MKL |

For further details, see [**III - Linear Algebra Benchmark**](docs/III_benchmark.md).

## VI - Regression Algorithms

**In progress...**