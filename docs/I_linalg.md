# I - Linear Algebra
Before implementing ML and regression algorithms, we need fundamental linear algebra operations and functions. It's the core of the optimization process to get better performances, so we'll try 5 different approachs to compare: 

### 1 - Naive
Basic C++ implementation using standard loops and operations. Still with basic optimization: column-major storage for cache-friendly operations. Serves as baseline for performance comparison.

### 2 - AVX2
Implementation using AVX2 SIMD instructions for vectorized operations, processing 256-bit registers (4 doubles simultaneously) and operating on blocks where applicable.

### 3 - AVX2 Threaded
Multi-threaded AVX2 implementation combining SIMD vectorization with parallel processing across CPU cores using std::thread and operating on blocks where applicable for maximum performance.

### 4 - Eigen
C++ template library for linear algebra: matrices, vectors, numerical solvers and related algorithms. For further details, see https://gitlab.com/libeigen/eigen.

### 5 - MKL
Intel's optimized Math Kernel Library for high-performance computing. For further details, see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html.

#### Note: 
By default, the backend used will be the best performing one among the three customized implementations, excluding MKL and Eigen libraries. Moreover, to compile MKL, the option MSVC is available to avoid any issue with MinGW-GCC.