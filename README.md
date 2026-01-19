# Machine Learning From Scratch
Educational project implementing fundamental machine learning and regression algorithms in C++ without relying on external ML libraries.

## I - Linear Algebra
Before implementing ML and regression algorithms, we need fundamental linear algebra operations and functions. It's the core of the optimization process to get better performances, so we'll try 5 different approachs to compare: 

###  Naive
Basic C++ implementation using standard loops and operations. Still with basic optimization: column-major storage for cache-friendly operations. Serves as baseline for performance comparison.

### AVX2
Implementation using AVX2 SIMD instructions for vectorized operations, processing 256-bit registers (4 doubles simultaneously) and operating on blocks where applicable.

### AVX2 Threaded
Multi-threaded AVX2 implementation combining SIMD vectorization with parallel processing across CPU cores using std::thread and operating on blocks where applicable for maximum performance.

### Eigen
C++ template library for linear algebra: matrices, vectors, numerical solvers and related algorithms. For further details, see https://gitlab.com/libeigen/eigen.

### MKL
Intel's optimized Math Kernel Library for high-performance computing. For further details, see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html.

#### Note: 
By default, the backend used will be the best performing one among the three customized implementations, excluding MKL and Eigen libraries. Moreover, to compile MKL, the option MSVC is available to avoid any issue with MinGW-GCC.

## II - Linear Algebra Benchmark
Comparison between backends using Google Benchmark across 4 operations

**Test Environment:**
- **CPU:** Intel Core i5-10300H @ 2.50GHz (4C/8T, Comet Lake)
- **RAM:** 8GB DDR4-2933
- **Compiler:** MSVC 2022 (`/O2 /arch:AVX2`)
- **OS:** Windows 11
- **Matrix sizes:** 64×64 to 4096×4096
- **Precision:** Double (FP64)

### 1) - Transpose
Matrix transpose swaps rows and columns (Aᵀ[i,j] = A[j,i]). Memory-bound, O(n²).

**Implementation details:**
- **Naive:** Copy elements to new matrix
- **AVX2 / AVX2 Threaded Blocked:** Cache-optimized block transpose 4x4 blocks using SIMD instructions to process 4 doubles simultaneously
- **Eigen/MKL:** Library-specific optimizations

The blocked approach improves cache hit rates by keeping working sets in L1/L2 cache.

<p align="center">
  <img src="benchmark/transpose.png" width="600">
</p>

Apart from Naive and Eigen, **backends show equivalent performance**. All backends saturate the available memory bandwidth (max transfer speed between RAM and CPU).

**Notes:**
- **AVX2TH** = AVX2 with multithreading
- **Naive:** Benchmark stopped at n=1024 due to catastrophic performance
- **Eigen:** Severe degradation at n≥1024, likely caused by double memory copy

### 2) - In-place Transpose
Swaps rows and columns within the same matrix (no allocation). Memory-bound, O(n²).

**Implementation details:**
- **Naive:** Direct element swapping
- **AVX2 / AVX2 Threaded Blocked:** Cache-blocked algorithm processing symmetric pairs:
  - Load blocks A[i,j] and A[j,i] (transposed positions)
  - Transpose both blocks
  - Swap and write back
- **Eigen/MKL:** Library-specific optimizations

This symmetric approach processes two blocks simultaneously, improving cache efficiency.

<p align="center">
  <img src="benchmark/transposeinplace.png" width="600">
</p>

**AVX2 Threaded dominates**. It might be due to Eigen and MKL being single threaded on in-place transpose. As we can see, with 4 threads, AVX2 Threaded is 4x faster, so we can make the assumption.

**Notes:**
- **AVX2TH** = AVX2 with multithreading
- **Naive:** Benchmark stopped at n=1024 due to catastrophic performance

### 3) - Matrix multiplication
Compute-intensive, O(n³)
<p align="center">
  <img src="benchmark/multiply.png" width="600">
</p>

### 4) - Inverse
Mixed (compute + memory), O(n³)
<p align="center">
  <img src="benchmark/inverse.png" width="600">
</p>