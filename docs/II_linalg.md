# II - Linear Algebra
Before implementing any ML and regression algorithms, we need fundamental linear algebra operations and functions. It's the core of the optimization process to achieve better performance, so we will implement 3 different approaches and add 2 external libraries to compare:

## Implemented
### Naive
Basic C++ implementation using standard loops and operations. Still with basic optimization: column-major storage for cache-friendly operations. Serves as a baseline for performance comparison.

### AVX2
Implementation using AVX2 SIMD instructions for vectorized operations, processing 256-bit registers (4 doubles simultaneously) and operating on blocks where applicable.

### AVX2 Threaded
Multi-threaded AVX2 implementation combining SIMD vectorization with parallel processing across CPU cores using std::thread and operating on blocks where applicable for maximum performance.

## External
### Eigen 
C++ template library for linear algebra: matrices, vectors, numerical solvers and related algorithms. For further details, see https://gitlab.com/libeigen/eigen.

### MKL
Intel's optimized Math Kernel Library for high-performance computing. For further details, see https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html.

#### Note: 
By default, the backend used will be the best performing one among the three customized implementations, excluding MKL and Eigen libraries. Moreover, to compile MKL, the MSVC option is available to avoid any issue with MinGW-GCC.

To read the next part: [**III - Linear Algebra Benchmark**](docs/III_benchmark.md).
