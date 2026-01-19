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
**Comparison between backends using Google Benchmark across 4 operations:**
- **Transpose** - Memory-bound, O(n²)
- **In-place Transpose** - Memory-bound, O(n²)
- **Matrix multiplication** - Compute-intensive, O(n³)
- **Inverse** - Mixed (compute + memory), O(n³)

**Test Environment:**
- **CPU:** Intel Core i5-10300H @ 2.50GHz (4C/8T, Comet Lake)
- **RAM:** 8GB DDR4-2933
- **Compiler:** MSVC 2022 (`/O2 /arch:AVX2`)
- **OS:** Windows 11 x64
- **Matrix sizes:** 64×64 to 4096×4096
- **Precision:** Double (FP64)

### 1) - Transpose
<p align="center">
  <img src="benchmark/transpose.png" width="600">
</p>

### 2) - In-place Transpose
<p align="center">
  <img src="benchmark/transposeinplace.png" width="600">
</p>

### 3) - Matrix multiplication
<p align="center">
  <img src="benchmark/multiply.png" width="600">
</p>

### 4) - Inverse
<p align="center">
  <img src="benchmark/inverse.png" width="600">
</p>