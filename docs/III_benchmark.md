# III - Linear Algebra Benchmark
This benchmark section aims to evaluate and compare the performance of our custom-implemented backends (`Naive`, `AVX2`, `AVX2TH`) against industry-standard production libraries (`MKL`, `Eigen`) across a range of matrix sizes and operations using Google Benchmark. <br>
Firstly, let's take a look at our environment: 

**Environment:**
- **CPU:** Intel Core i5-10300H @ 2.50GHz (4C/8T, Comet Lake)
- **RAM:** 8GB DDR4-2933
- **Compiler:** MSVC 2022 (`/O2 /arch:AVX2`)
- **OS:** Windows 11
- **Matrix sizes:** 64×64 to 4096×4096
- **Precision:** Double (FP64)

**Legend:** 
- **`AVX2TH`** = **`AVX2`** with multithreading
- **`Naive`:** Benchmark stopped at n=1024 due to excessive runtime

## 1 - Transpose
Matrix transpose swaps rows and columns (Aᵀ[i,j] = A[j,i]). Memory-bound, O(n²).

- **`Naive`:** Copy elements to a new matrix
- **`AVX2` / `AVX2 Threaded`:** Cache-optimized block transpose 4x4 blocks using SIMD instructions to process 4 doubles simultaneously

Implementations use blocking to keep data in L1/L2 cache and leverage SIMD for parallel arithmetic.

<p align="center">
  <img src="assets/transpose.png" width="600">
</p>

Apart from `Naive` and `Eigen`, **backends show equivalent performance**. All backends saturate the available memory bandwidth (max transfer speed between RAM and CPU).

Note: **`Eigen`** degrades at n≥1024, likely caused by double memory copy

## 2 - In-place Transpose
Swaps rows and columns within the same matrix (no allocation). Memory-bound, O(n²).

- **`Naive`:** Direct element swapping
- **`AVX2` / `AVX2 Threaded`:** Cache-blocked algorithm processing symmetric pairs:
  - Load blocks A[i,j] and A[j,i] (transposed positions)
  - Transpose both blocks
  - Swap them

This symmetric approach processes two blocks simultaneously, improving cache efficiency.

<p align="center">
  <img src="assets/transposeinplace.png" width="600">
</p>

**`AVX2 Threaded` outperforms**. `Eigen` and `MKL` might be single threaded on in-place transpose operation. As we can see, with 4 threads, `AVX2 Threaded` is 4x faster, so we can make the assumption.

## 3 - Matrix multiplication
Computes C = A × B where all matrices are n×n. Compute-bound, O(n³).

- **`Naive`:** Triple nested loop (i, j, k order)
- **`AVX2` / `AVX2 Threaded`:** Cache-blocked algorithm with:
  - Vectorized inner loops using AVX2 FMA instructions
  - Optimized memory access patterns to maximize cache hits <br> 

<p align="center">
  <img src="assets/multiply.png" width="600">
</p>

**`MKL` and `Eigen` outperform as expected**. `AVX2 Threaded` shows decent scaling with 4 threads but cannot match the highly tuned kernels of `MKL`/`Eigen` for this compute-intensive operation.

## 4 - Inverse
Computes A⁻¹ for an n×n matrix. Mixed (compute + memory), O(n³)

**`Naive` / `AVX2` / `AVX2 Threaded` with LU decomposition approach:**
- Decompose A = LU with partial pivot
- Solve LUX = I by forward/backward substitution
- Result X = A⁻¹

LU-based inversion involves both matrix decomposition (compute-intensive) and triangular solves (memory access patterns matter).

<p align="center">
  <img src="assets/inverse.png" width="600">
</p>

**`MKL` and `Eigen` vastly outperform `AVX2 Threaded`** due to superior decomposition algorithms.

**Note:** **`AVX2`** stopped at n=1024 due to excessive runtime

## Key takeaways

- **Multithreading** provides meaningful gains (~4×) but only when the operation scales well across threads
- **Memory bandwidth saturation** is the primary bottleneck for O(n²) operations, limiting the benefit of further optimization
- **Custom AVX2 implementations** are a viable option for memory-bound workloads but for production use, compute-intensive operations should be implemented with libraries like MKL or Eigen 
<br> <br>

**To test it yourself**, you can also check the corresponding files: 
- [**bench_data.cpp**](/benchmark/bench_data.cpp)
- [**bench_linalg.cpp**](/benchmark/bench_linalg.cpp)
<br>

#### Note: 
By default, the backend used will be the best performing one among the three customized implementations, excluding MKL and Eigen libraries. Moreover, to compile MKL, the MSVC option is available to avoid any issue with MinGW-GCC.

To read the next part: [**VI - Statistical Functions**](/docs/VI_stats.md).